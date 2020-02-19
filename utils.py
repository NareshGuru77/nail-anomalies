import tensorflow as tf


def lr_schedule(schedule_params):
    if not type(schedule_params) is dict:
        return schedule_params
    known_lr_schedules = {'cosine_decay': tf.train.cosine_decay,
                          'cosine_restarts': tf.train.cosine_decay_restarts,
                          'exponential': tf.train.exponential_decay}
    lr_type = schedule_params.pop('type')
    if lr_type == 'constant':
        return schedule_params['learning_rate']
    assert lr_type in known_lr_schedules.keys(), 'Unknown lr schedule'
    lr_s = known_lr_schedules[lr_type](
        global_step=tf.train.get_global_step(), **schedule_params)
    tf.summary.scalar('learning rate', lr_s)
    return lr_s
