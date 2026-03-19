"""Dark Photon Dataset."""

import glob
import math
import os
import os.path as osp
import shutil
import tarfile

import deepdish as dd
import torch
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm


def is_noise_subdir(subdir):
    """Check if a subdirectory is noise or signal.

    Parameters
    ----------
    subdir : str
        Subdirectory name.

    Returns
    -------
    bool
        Whether the subdirectory is noise or signal.
    """
    return "background" in subdir.lower()


class DarkPhotonDataset(InMemoryDataset):
    """Dataset containing samples of jetgraphs.

    The downloaded dataset will be built with fully connected edges and edge features.

    Parameters
    ----------
    root : str
        Where to download (or look for) the dataset.
    url : str
        URL to the dataset.
    subset : str
        Percentage of dataset to be used. Default is 1.0.
    verbose : bool
        Whether to display more info while processing graphs.
    transform : callable
        A function/transform that takes in an `torch_geometric.data.Data` object and returns a transformed version.
    pre_transform : callable
        A function/transform that takes in an `torch_geometric.data.Data` object and returns a transformed version.
    pre_filter : callable
        A function that takes in an `torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset.
    post_filter : callable
        A function that takes in an `torch_geometric.data.Data` object and returns a boolean value, indicating whether the data object should be included in the final dataset.
    remove_download : bool
        Whether to remove the compressed download file after extraction. Default is False.
    **kwargs
        Additional arguments.
    """

    def __init__(
        self,
        root,
        url="https://cernbox.cern.ch/s/PYurUUzcNdXEGpz/download",
        subset=1.0,
        verbose=False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        post_filter=None,
        remove_download=False,
        **kwargs,
    ):
        self.url = url
        self.subset = subset
        self.verbose = verbose
        self.post_filter = post_filter
        self.remove_download = remove_download
        self.kwargs = kwargs

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices, subset = torch.load(
            self.processed_paths[0], weights_only=False
        )

        print(f"Loaded dataset containing subset of {subset}")
        if subset != self.subset:
            print(
                "The loaded dataset has different settings from the ones requested. Processing graphs again."
            )
            self.process()
            self.data, self.slices, subset = torch.load(
                self.processed_paths[0], weights_only=False
            )

    def __repr__(self):
        """String representation of the dataset.

        Returns
        -------
        str
            String representation of the dataset.
        """
        return f"{self.__class__.__name__}({len(self)})"

    def download(self):
        """Download dataset.

        Download tar file to raw_dir, extract directories to raw_dir and rename files.
        Finally cleanup all unused files.
        """
        # Download self.url to self.raw_dir.
        download_url(self.url, self.raw_dir)
        os.rename(
            osp.join(self.raw_dir, "download"),
            osp.join(self.raw_dir, "download.tar"),
        )

        # Extract everything in self.raw_dir.
        with tarfile.open(osp.join(self.raw_dir, "download.tar")) as tar:
            tar.extractall(self.raw_dir)
            tar.close()

        # Clean.
        if self.remove_download:
            print("Removing compressed download...")
            os.remove(osp.join(self.raw_dir, "download.tar"))

        # Rename files.
        self._rename_filenames()

    def _rename_filenames(self):
        """Rename files.

        Filenames are poorly named and so we have to rename them for easier processing.
        This function just renames all downloaded files to make the processing clear.
        """

        print("Moving Directories to raw directory...")
        pattern = os.path.join(self.raw_dir, "**", "*_v6")
        data_dirs = glob.glob(
            pattern, recursive=True
        )  # should return Signal_v6, Background2_v6, Background3_v6
        for subdir in data_dirs:
            shutil.move(subdir, self.raw_dir)

        print("Renaming files...")
        # Add the 'a0' prefix to files for layer 0.
        for c in ["s", "b"]:  # signal, background
            # match any .h5 file in self.raw_dir that does not start with 'a0'.
            pattern = os.path.join(self.raw_dir, "*", f"[{c}]*.h5")
            result = glob.glob(pattern)
            for filename in result:
                # add a0 which is missing in all directories
                os.rename(
                    filename,
                    filename.replace(f"{c}6", f"a0_{c}6").replace(f"{c}6", ""),
                )

        # Remove useless version number and make file names prettier.
        for c in ["s", "b"]:  # signal, background
            # match any .h5 file in self.raw_dir and remove useless version number.
            pattern = os.path.join(self.raw_dir, "*", "*.h5")
            result = glob.glob(pattern)
            for filename in result:
                os.rename(filename, filename.replace(f"{c}6", ""))
        print("Done renaming files!")

    def _preprocess(self):
        """Preprocessing of the data.

        Preprocess data to build dataset of fully connected graphs with edge features, then save it.
        """
        print("[Preprocessing] Preprocessing data to build fully connected graphs.")

        if not os.path.exists(os.path.join(self.raw_dir, "Signal_v6")):
            print("Probably something went wrong during download. Trying to rename files again...")
            self._rename_filenames()

        signal_list = []
        background_list = []
        # Attributes to retrieve for each graph.
        attributes = ["eta", "phi", "energy", "energyabs"]

        # Process each subdirectory separately.
        pattern = os.path.join(self.raw_dir, "*_v6")
        data_dirs = glob.glob(pattern)
        print(data_dirs)
        for subdir in data_dirs:
            print(f"[Preprocessing] Reading files in {subdir}...")
            is_noise = is_noise_subdir(subdir)

            # Build dictionary of raw data from each subdirectory independently.
            dataset_by_layer = {}
            for layer in range(4):
                dataset_by_layer[layer] = {}
                for attribute in attributes:
                    filepath = os.path.join(subdir, f"a{layer}_tupleGraph_bar{attribute}.h5")
                    if self.verbose:
                        print(f"[Preprocessing] Reading: {filepath}")
                    dataset_by_layer[layer][attribute] = dd.io.load(filepath)

            num_graphs = len(dataset_by_layer[0][attributes[0]])
            # Process raw tuples contained in dictionary.
            print("[Preprocessing] Building graphs...")
            for gid in tqdm(range(1, num_graphs)):
                _nodes = []
                for layer in range(4):
                    _layer_nodes = []
                    for attribute in attributes:
                        nodes = torch.tensor(
                            list(dataset_by_layer[layer][attribute][gid].values())
                        )
                        nodes_to_fix = nodes > math.pi
                        nodes[nodes_to_fix] -= 2 * math.pi
                        nodes_to_fix = nodes < -math.pi
                        nodes[nodes_to_fix] += 2 * math.pi
                        _layer_nodes.append(nodes)
                    _layer_nodes.insert(0, torch.ones_like(_layer_nodes[-1]) * layer)
                    layer_nodes = torch.cat(_layer_nodes, dim=-1)
                    _nodes.append(layer_nodes)

                nodes = torch.cat(_nodes, dim=0)

                # Filter out nodes based on conditions.
                invalid_nodes_mask = nodes[:, -1] <= 400
                nodes = nodes[~invalid_nodes_mask]

                # If no nodes are left after deleting unwanted, just skip this graph.
                if nodes.shape[0] == 0:
                    continue

                # Define edges (edge_index) for a fully connected graph.
                num_nodes = nodes.size(0)
                edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Add reverse edges for undirected graph

                # Define edge features (edge_attr) as the Euclidean distance between connected nodes.
                edge_attr = torch.norm(nodes[edge_index[0]] - nodes[edge_index[1]], dim=1, keepdim=True)

                # Finally create and append graph to list.
                graph_class = 0 if is_noise else 1

                # Last column is absolute energy, not useful from now, so we remove it.
                nodes = nodes[:, :-1]
                graph = Data(
                    x=nodes,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    y=graph_class
                )
                if is_noise:
                    background_list.append(graph)
                else:
                    signal_list.append(graph)

            print(f"[Preprocessing] Done preprocessing files in {subdir}")

        print("[Preprocessing] Done preprocessing all subdirectories!")

        # Save obtained torch tensor for later.
        torch.save([signal_list, background_list], self.pre_processed_path)

    def process(self):
        """Processing dataset.

        Process dataset to build graphs with edges, then save it.
        """
        # If raw data was not preprocessed, do it now.
        if not os.path.exists(self.pre_processed_path):
            self._preprocess()

        # Load preprocessed graphs.
        print("[Processing] Loading preprocessed data...")
        [signal_list, background_list] = torch.load(
            self.pre_processed_path, weights_only=False
        )

        # Work out amount of graphs to process.
        signal_graphs = int(len(signal_list) * self.subset)
        noise_graphs = int(len(background_list) * self.subset)
        # Build (possibly reduced) list of graphs.
        signal_subset = signal_list[:signal_graphs]
        noise_subset = background_list[:noise_graphs]
        processed_data_list = signal_subset + noise_subset

        if self.pre_filter is not None:
            print("[Processing] Pre-filtering out unwanted graphs...")
            processed_data_list = [
                data
                for data in tqdm(processed_data_list)
                if self.pre_filter(data)
            ]

        if self.pre_transform is not None:
            print("[Processing] Applying pre transform...")
            processed_data_list = [
                self.pre_transform(data) for data in tqdm(processed_data_list)
            ]

        if self.post_filter is not None:
            print("[Processing] Post-filtering out unwanted graphs...")
            processed_data_list = [
                data
                for data in tqdm(processed_data_list)
                if self.post_filter(data)
            ]

        # Save obtained torch tensor.
        data, slices = self.collate(processed_data_list)
        torch.save((data, slices, self.subset), self.processed_paths[0])

    # AUXILIARY FUNCTIONS
    def stats(self):
        """Print dataset statistics."""
        print("\n*** JetGraph Dataset ***\n")
        print(f"Number of classes: {self.num_classes}")
        print(f"Number of graphs: {len(self)}")
        print(f"Dataset is directed: {self.is_directed}")
        print(f"Number of node features: {self.num_node_features}")
        print(f"Number of edge features: {self.num_edge_features}")
        print(f"Number of positive samples:{self.num_positive_samples:.2f}")

    # PROPERTIES
    @property
    def is_directed(self):
        """Whether the dataset is directed.

        Returns
        -------
        bool
            Whether the dataset is directed.
        """
        return self.get(0).is_directed()

    @property
    def num_node_features(self) -> int:
        """Number of node features in the dataset.

        Returns
        -------
        int
            Number of node features in the dataset.
        """
        return self.get(0).x.size(1)

    @property
    def num_edge_features(self) -> int:
        """Number of edge features in the dataset.

        Returns
        -------
        int
            Number of edge features in the dataset.
        """
        sample = self.get(0).edge_attr
        return sample.size(1) if sample is not None else None

    @property
    def num_positive_samples(self):
        """Number of positive samples in the dataset.

        Returns
        -------
        int
            Number of positive samples in the dataset.
        """
        return sum([x.y.item() for x in self])

    @property
    def raw_file_names(self):
        """List of raw files.

        Returns
        -------
        list
            List of raw files.
        """
        return ["Background2_v6", "Background3_v6", "Signal_v6"]

    @property
    def processed_file_names(self):
        """List of processed files.

        Returns
        -------
        list
            List of processed files.
        """
        return ["jet_graph_processed.pt"]

    @property
    def pre_processed_path(self):
        """Path to preprocessed data.

        Returns
        -------
        str
            Path to preprocessed data.
        """
        return os.path.join(self.raw_dir, "preprocessed_no_edges.list")
