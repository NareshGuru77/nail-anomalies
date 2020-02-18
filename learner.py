from tensorflow.estimator import Estimator
from dataset.nails_dataset import NailsDataset
import tensorflow as tf

from models.cnn import model_fn


class Learner(Estimator):

    def __init__(self, model_dir, **kwargs):
        self.train_files = kwargs['train_files']
        self.train_transforms = kwargs['train_transforms']
        self.test_files = kwargs['test_files']
        self.test_transforms = kwargs['test_transforms']
        self.image_size = kwargs['image_size']
        self.train_dataloading = kwargs['train_dataloading']
        self.test_dataloading = kwargs['test_dataloading']
        self.train_max_steps = kwargs['train_max_steps']

        warm_start_from = kwargs.get('warm_start_from', None)
        vars_to_warm_start = kwargs.get('vars_to_warm_start', '.*')
        if warm_start_from is not None:
            self.ws = tf.estimator.WarmStartSettings(
                ckpt_to_initialize_from=warm_start_from,
                vars_to_warm_start=vars_to_warm_start)
        else:
            self.ws = None

        self.save_checkpoints_steps = kwargs.get(
            'save_checkpoints_steps', 1000)
        save_summary_steps = kwargs.get('save_summary_steps', 1000)
        tf_random_seed = kwargs.get('tf_random_seed', None)
        self.model_params = kwargs['model_params']

        allow_growth = kwargs.get('allow_growth', False)
        config = None
        if allow_growth:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
        run_config = tf.estimator.RunConfig(
            tf_random_seed=tf_random_seed,
            model_dir=model_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            save_summary_steps=save_summary_steps,
            session_config=config)

        super(Learner).__init__(model_fn, model_dir, run_config,
                                self.model_params, self.ws)

        self.iterator = None
        self.output_shapes = {'image': tf.TensorShape([self.image_size[0],
                                                       self.image_size[1], 3])}, \
                             {'label': tf.TensorShape([])}
        self.output_types = {'image': tf.float32}, {'label': tf.int32}

    def eval_exporters(self):
        best_exporter = tf.estimator.BestExporter(
            name='best_exporter',
            serving_input_receiver_fn=self.serving_input_receiver_fn,
            compare_fn=self.ap_higher)
        latest_exporter = tf.estimator.LatestExporter(
            name='latest_exporter',
            serving_input_receiver_fn=self.serving_input_receiver_fn)

        return [best_exporter, latest_exporter]

    def train_dataset(self):
        return NailsDataset(self.train_files, self.train_transforms)

    def get_train_kwargs(self):
        return {'input_fn': self.create_input_fn(), 'hooks': [],
                'max_steps': self.train_max_steps}

    def test_dataset(self):
        return NailsDataset(self.test_files, self.test_transforms)

    def get_eval_kwargs(self, checkpoint_path):
        return {'input_fn': self.create_input_fn(is_train=False),
                'steps': len(self.test_dataset()), 'hooks': [],
                'checkpoint_path': checkpoint_path}

    def get_train_and_eval_kwargs(self, start_delay_secs=120,
                                  throttle_secs=600):
        input_fn = self.create_input_fn()
        train_spec = tf.estimator.TrainSpec(
            input_fn=input_fn, max_steps=self.max_steps,
            hooks=self.get_train_hooks())

        input_fn = self.create_input_fn(is_train=False)
        eval_spec = tf.estimator.EvalSpec(
            input_fn=input_fn, hooks=self.get_eval_hooks(),
            exporters=self.get_eval_exporters(), steps=len(self.test_dataset()),
            start_delay_secs=start_delay_secs, throttle_secs=throttle_secs)

        return {'estimator': self, 'train_spec': train_spec,
                'eval_spec': eval_spec}

    @staticmethod
    def serving_input_receiver_fn():
        image = tf.placeholder(shape=[None, None, None, 3],
                               dtype=tf.float32, name='image')
        data = {'image': image}
        return tf.estimator.export.ServingInputReceiver(
            features=data, receiver_tensors=data)

    def generator(self):
        while True:
            try:
                idx = next(self.iterator)
                data, label = self.__getitem__(idx)
                yield data, label
            except StopIteration:
                break

    def create_input_fn(self, is_train=True):
        def input_fn():
            dataset = tf.data.Dataset.from_generator(
                self.get_generator, output_shapes=self.output_shapes,
                output_types=self.output_types)

            if 'buffer_size' in shuffle_params.keys():
                dataset = dataset.shuffle(**shuffle_params)
            if repeat:
                dataset = dataset.repeat()
            if 'batch_size' in batch_params.keys():
                dataset = dataset.batch(**batch_params)
            if 'buffer_size' in prefetch_params.keys():
                dataset = dataset.prefetch(**prefetch_params)

            return dataset

        if is_train:
            dataset = self.train_dataset()
            dataloading = self.train_dataloading
        else:
            dataset = self.test_dataset()
            dataloading = self.test_dataloading

        self.iterator = iter(dataset)
        shuffle_params = dataloading.get('shuffle_params', {})
        batch_params = dataloading.get('batch_params', {})
        prefetch_params = dataloading.get('prefetch_params', {})
        repeat = dataloading.get('repeat', False)

        return input_fn
