import dill



def _build_train_loader(self):
    config = self.config
    self.train_scenes = []

    with open(self.train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    # # 这个循环的目的是通过配置文件修改两个agent之间的attention_radius的
    # for attention_radius_override in config.override_attention_radius:
    #     node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
    #     train_env.attention_radius[(node_type1, node_type2)] = float(attention_radius)


    self.train_scenes = self.train_env.scenes
    self.train_scenes_sample_probs = self.train_env.scenes_freq_mult_prop if config.scene_freq_mult_train else None

    self.train_dataset = EnvironmentDataset(train_env,
                                        self.hyperparams['state'],
                                        self.hyperparams['pred_state'],
                                        scene_freq_mult=self.hyperparams['scene_freq_mult_train'],
                                        node_freq_mult=self.hyperparams['node_freq_mult_train'],
                                        hyperparams=self.hyperparams,
                                        min_history_timesteps=1,
                                        min_future_timesteps=self.hyperparams['prediction_horizon'],
                                        return_robot=not self.config.incl_robot_node)
    self.train_data_loader = dict()
    for node_type_data_set in self.train_dataset:
        node_type_dataloader = utils.data.DataLoader(node_type_data_set,
                                                        collate_fn=collate,
                                                        pin_memory = True,
                                                        batch_size=self.config.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.config.preprocess_workers)
        self.train_data_loader[node_type_data_set.node_type] = node_type_dataloader
        
class EnvironmentDataset(object):
    def __init__(self, env, state, pred_state, node_freq_mult, scene_freq_mult, hyperparams, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']
        self.node_type_datasets = list()
        self._augment = False
        for node_type in env.NodeType:
            if node_type not in hyperparams['pred_state']:
                continue
            self.node_type_datasets.append(NodeTypeDataset(env, node_type, state, pred_state, node_freq_mult,
                                                           scene_freq_mult, hyperparams, **kwargs))

    @property
    def augment(self):
        return self._augment

    @augment.setter
    def augment(self, value):
        self._augment = value
        for node_type_dataset in self.node_type_datasets:
            node_type_dataset.augment = value

    def __iter__(self):
        return iter(self.node_type_datasets)


class NodeTypeDataset(data.Dataset):
    def __init__(self, env, node_type, state, pred_state, node_freq_mult,
                 scene_freq_mult, hyperparams, augment=False, **kwargs):
        self.env = env
        self.state = state
        self.pred_state = pred_state
        self.hyperparams = hyperparams
        self.max_ht = self.hyperparams['maximum_history_length']
        self.max_ft = kwargs['min_future_timesteps']

        self.augment = augment

        self.node_type = node_type
        self.index = self.index_env(node_freq_mult, scene_freq_mult, **kwargs)
        self.len = len(self.index)
        self.edge_types = [edge_type for edge_type in env.get_edge_types() if edge_type[0] is node_type]

    def index_env(self, node_freq_mult, scene_freq_mult, **kwargs):
        index = list()
        for scene in self.env.scenes:
            present_node_dict = scene.present_nodes(np.arange(0, scene.timesteps), type=self.node_type, **kwargs)
            for t, nodes in present_node_dict.items():
                for node in nodes:
                    index += [(scene, t, node)] *\
                             (scene.frequency_multiplier if scene_freq_mult else 1) *\
                             (node.frequency_multiplier if node_freq_mult else 1)

        return index

    def __len__(self):
        return self.len

    def __getitem__(self, i):
        (scene, t, node) = self.index[i]

        if self.augment:
            scene = scene.augment()
            node = scene.get_node_by_id(node.id)

        return get_node_timestep_data(self.env, scene, t, node, self.state, self.pred_state,
                                      self.edge_types, self.max_ht, self.max_ft, self.hyperparams)