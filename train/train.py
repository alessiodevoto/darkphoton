import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score
import os

# Early stopping class to prevent overfitting and stop training when validation loss stops improving
class EarlyStopping():
    def __init__(self, patience, delta=0, path='checkpoint.pt'):
        self.patience = patience  # Number of epochs to wait before stopping after no improvement
        self.delta = delta  # Minimum change in loss to qualify as an improvement
        self.path = path  # Path to save the best model
        self.counter = 0  # Counts the number of epochs with no improvement
        self.best_score = None  # Tracks the best validation score
        self.early_stop = False  # Flag to indicate if training should stop

    def __call__(self, val_loss, model):
        score = -val_loss  # Convert loss to a score (minimization problem)
        
        # Initialize best score in the first call
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score - self.delta:
            # No improvement, increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True  # Stop training
        else:
            # Improvement found, save model and reset counter
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Save the model when validation loss decreases."""
        torch.save(model.state_dict(), self.path)
        print(f'Model saved with validation loss: {val_loss}')


# Training function
def train(dataloader, model, loss_fn, optimizer, device):
    model.train()  # Set model to training mode
    predictions = []
    ground_truths = []
    losses = []
    expert_sample_counts = []  # Tensor of shape (num_experts) tracking number of samples processed by each expert
    expert_loads = []  # Load distribution tensor of shape (num_experts)

    for batch in dataloader:
        optimizer.zero_grad()  # Reset gradients
        prediction = model(batch.to(device))  # Forward pass
        batch_loss = loss_fn(prediction, batch.y)  # Compute loss
        
        batch_loss.backward()  # Backpropagation
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        optimizer.step()  # Update model weights

        # Store results for analysis
        predictions.append(prediction[0])
        ground_truths.append(batch.y)
        losses.append(batch_loss)
        expert_sample_counts.append(prediction[2])
        expert_loads.append(prediction[1])

    return predictions, ground_truths, expert_loads, expert_sample_counts


# Evaluation function (similar to training but without gradient updates)
def evaluate(dataloader, model, loss_fn, device):
    model.eval()  # Set model to evaluation mode
    predictions = []
    ground_truths = []
    losses = []
    expert_loads = []  # Load distribution tensor of shape (num_experts)
    expert_sample_counts = []  # Tensor of shape (num_experts) tracking samples processed
    node_expert_assignments = []  # Tensor tracking node-to-expert assignment across layers

    with torch.no_grad():  # Disable gradient computation for efficiency
        for batch in dataloader:
            prediction = model(batch.to(device))  # Forward pass
            batch_loss = loss_fn(prediction, batch.y)  # Compute loss

            # Store results
            predictions.append(prediction[0])
            ground_truths.append(batch.y)
            losses.append(batch_loss)
            expert_loads.append(prediction[1])
            expert_sample_counts.append(prediction[2])
            node_expert_assignments.append(prediction[3])

    return predictions, ground_truths, expert_loads, expert_sample_counts, node_expert_assignments


# Training and evaluation loop
def train_evaluate(trainloader, evalloader, model, criterion, loss_fn, optimizer, scheduler, patience, epochs, device, save_dir="training_outputs"):
    os.makedirs(save_dir, exist_ok=True)  # Create a folder to save outputs

    train_history, eval_history = [], []
    early_stopping = EarlyStopping(patience=epochs)  # Initialize early stopping
    scheduler_wait_counter = 0

    # Metrics tracking
    train_accuracy_list, train_precision_list, train_recall_list, train_load_balance_list = [], [], [], []
    eval_accuracy_list, eval_precision_list, eval_recall_list, eval_load_balance_list = [], [], [], []

    for epoch in range(epochs):
        running_loss = 0.0

        # Perform training and evaluation
        train_results = train(dataloader=trainloader, model=model, loss_fn=loss_fn, optimizer=optimizer, device=device)
        eval_results = evaluate(dataloader=evalloader, model=model, loss_fn=loss_fn, device=device)

        # Store history
        train_history.append(train_results)
        eval_history.append(eval_results)

        # Learning rate scheduling (if applicable)
        if scheduler is not None:
            if scheduler_wait_counter == patience:
                model.load_state_dict(torch.load('checkpoint.pt'))  # Load best model
                print("Best model loaded before stepping")
            scheduler.step()
        scheduler_wait_counter += 1

        # Compute overall losses for training and evaluation
        train_loss = criterion(torch.cat(train_results[0]), torch.cat(train_results[1]))
        eval_loss = criterion(torch.cat(eval_results[0]), torch.cat(eval_results[1]))

        # Convert predictions to class labels
        train_preds_epoch = torch.argmax(torch.cat(train_results[0]), dim=1)
        eval_preds_epoch = torch.argmax(torch.cat(eval_results[0]), dim=1)

        # Compute classification metrics
        train_accuracy_list.append(accuracy_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_accuracy_list.append(accuracy_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        train_precision_list.append(precision_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_precision_list.append(precision_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        train_recall_list.append(recall_score(torch.cat(train_results[1]).cpu(), train_preds_epoch.cpu()))
        eval_recall_list.append(recall_score(torch.cat(eval_results[1]).cpu(), eval_preds_epoch.cpu()))

        # Compute load balancing statistics
        train_load_balance_list.append(torch.mean(torch.tensor(train_results[3])))
        eval_load_balance_list.append(torch.mean(torch.tensor(eval_results[3])))

        # Print training progress
        print("Training loss:", train_loss, "Evaluation loss:", eval_loss)
        print("Epoch:", epoch + 1)

        # Check for early stopping
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    # Load the best model before returning
    model.load_state_dict(torch.load('checkpoint.pt'))

    # SAVE important things for plotting
    torch.save({
        "train_expert_sample_counts": train_results[3],
        "eval_expert_sample_counts": eval_results[3],
        "eval_node_expert_assignments": eval_results[4],
    }, os.path.join(save_dir, "experts_info.pt"))

    torch.save({
        "train_accuracy_list": train_accuracy_list,
        "train_precision_list": train_precision_list,
        "train_recall_list": train_recall_list,
        "train_load_balance_list": train_load_balance_list,
        "eval_accuracy_list": eval_accuracy_list,
        "eval_precision_list": eval_precision_list,
        "eval_recall_list": eval_recall_list,
        "eval_load_balance_list": eval_load_balance_list,
    }, os.path.join(save_dir, "metrics.pt"))


    return train_history, eval_history, (train_accuracy_list, train_precision_list, train_recall_list, train_load_balance_list), (eval_accuracy_list, eval_precision_list, eval_recall_list, eval_load_balance_list)
