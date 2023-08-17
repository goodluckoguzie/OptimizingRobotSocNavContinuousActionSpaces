
class Seq_Len:
    seq_len_1 = 2
    seq_1 = 'window_1'
    seq_len_4 = 5
    seq_4 = 'window_4'
    seq_len_8 = 9
    seq_8 = 'window_8'
    seq_len_6 = 7
    seq_6 = 'window_6'
    seq_len_16 = 17
    seq_16 = 'window_16_prediction_1'

    seq_len_99 = 100
    seq_99 = 'window_99_prediction_1'
    seq_len_50 = 51
    seq_50 = 'window_50_prediction_1'    
    seq_len_65 = 66
    seq_65 = 'window_65_prediction_1'
    seq_len_32 = 33
    seq_32 = 'window_32_prediction_1'
    seq_len_21 = 21
    seq_21 = 'window_16_prediction_5'
    seq_len_19 = 19
    seq_19 = 'window_16_prediction_3'
    seq_len_199 = 200
    seq_199 = 'window_199_prediction_1' 
    seq_len_349 = 350
    seq_349 = 'window_349_prediction_1'

class NonPrePaddedRobotFrame_Datasets_Timestep_0_5:
    data_dir = 'RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep05'

class NonPrePaddedRobotFrame_Datasets_Timestep_0_25:
    data_dir = 'RobotFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep025'

class NonPrePaddedWorldFrame_Datasets_Timestep_0_5:
    data_dir = 'WorldFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'NonPrePaddedWorldFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'NonPrePaddedWorldFrameDatasetsTimestep05'

class NonPrePaddedWorldFrame_Datasets_Timestep_0_25:
    data_dir = 'WorldFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'NonPrePaddedWorldFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'NonPrePaddedWorldFrameDatasetsTimestep025'

class NonPrePaddedRobotFrame_Datasets_Timestep_1:
    data_dir = 'RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep1'

class RobotFrameContinuousDatasetsTimestep1_15k:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1_15k'
    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1_15k'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1_15k'

class NonPrePaddedRobotFrame_Datasets_Timestep_2:
    data_dir = 'RobotFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'mainNonPrePaddedRobotFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'mainNonPrePaddedRobotFrameDatasetsTimestep2'



class WorldFrame_Datasets_Timestep_2:
    data_dir = 'WorldFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'WorldFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep2'


class WorldFrame_Datasets_Timestep_1:
    data_dir = 'WorldFrameDatasetsTimestep1'
    time_steps =  201
    RNN_runs = 'WorldFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep1'

class WorldFrame_Datasets_Timestep_0_5:
    data_dir = 'WorldFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'WorldFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep05'

class WorldFrame_Datasets_Timestep_0_25:
    data_dir = 'WorldFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'WorldFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'WorldFrameDatasetsTimestep025'


class RobotFrame_Datasets_Timestep_2:
    data_dir = 'RobotFrameDatasetsTimestep2'
    time_steps =  100
    RNN_runs = 'RobotFrameDatasetsTimestep2'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep2'

class RobotFrame_Continuous_Datasets_Timestep_1:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep1'

    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1'


class RobotFrame_Continuous_Datasets_Timestep_1x:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1x'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep1x'

    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1x'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1x'


class RobotFrame_Continuous_Datasets_Timestep_1s:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1s'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep1s'

    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1s'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1s'

class RobotFrame_Continuous_Datasets_Timestep_025:
    data_dir = 'RobotFrameContinuousDatasetsTimestep025'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep025'

    time_steps =  350
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep025'


class RobotFrame_Continuous_Datasets_Timestep_1b:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1b'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep1b'

    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1b'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1b'




class RobotFrame_Continuous_Datasets_Timestep_1b_500:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1b_500'
    data_dir500 = 'normalized_datasets'
    humanTrialData_dir = 'HumanTrialRobotFrameContinuousDatasetsTimestep1b'

    time_steps =  201
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1b_500'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1b_500'



class RobotFrameContinuousDatasetsTimestep1_15k:
    data_dir = 'RobotFrameContinuousDatasetsTimestep1_15k'
    data_dir500 = 'data_dir_normalized'
    humanTrialData_dir = 'RobotFrameContinuousDatasetsTimestep1_15k'

    time_steps =  200
    RNN_runs = 'RobotFrameContinuousDatasetsTimestep1_15k'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameContinuousDatasetsTimestep1_15k'

class NomalisedRobotFrameContinuousDatasets:
    data_dir = 'NomalisedRobotFrameContinuousDatasets'
    humanTrialData_dir = 'NomalisedRobotFrameContinuousDatasets'

    time_steps =  200
    RNN_runs = 'NomalisedRobotFrameContinuousDatasets'
    ckpt_dir = 'ckpt'
    rnnsave = 'NomalisedRobotFrameContinuousDatasets'



class RobotFrame_Datasets_Timestep_0_5:
    data_dir = 'RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'RobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep05'

class RobotFrame_Datasets_Timestep_0_25:
    data_dir = 'RobotFrameDatasetsTimestep025'
    time_steps =  800
    RNN_runs = 'RobotFrameDatasetsTimestep025'
    ckpt_dir = 'ckpt'
    rnnsave = 'RobotFrameDatasetsTimestep025'



class DQN_RobotFrame_Datasets_Timestep_1:
    data_dir = 'DQN_RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep1'

class DQN_RobotFrame_Datasets_Timestep_0_5:
    data_dir = 'DQN_RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep05'


class RndDQN_RobotFrame_Datasets_Timestep_1:
    data_dir = 'DQN_RobotFrameDatasetsTimestep1'
    time_steps =  200
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep1'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep1'

class RndDQN_RobotFrame_Datasets_Timestep_0_5:
    data_dir = 'DQN_RobotFrameDatasetsTimestep05'
    time_steps =  400
    RNN_runs = 'DQN_RobotFrameDatasetsTimestep05'
    ckpt_dir = 'ckpt'
    rnnsave = 'DQN_RobotFrameDatasetsTimestep05'


class HyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0




class WorldFrameHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    # data_dir = 'Datasetsworldframe'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    # seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 53 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0

class DQNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    controller = 'A3C'

    extra = False
    # data_dir = 'dqnDatasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'


    batch_size = 2 # actually batchsize * Seqlen
    seq_len = 10

    test_batch = 1
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    ctrl_hidden_dims = 512
    log_interval = 5000
    save_interval = 50

    use_binary_feature = False
    score_cut = 300 # to save
    save_start_score = 100

    # Rollout
    max_ep = 300
    n_rollout = 5000
    seed = 0

    n_workers = 0

class RNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    n_hiddens = 256
    extra = False
    # data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'
    seed = 0

    batch_size = 64 # actually batchsize * Seqlen
    test_batch = 1
    seq_len = 10
    n_sample = 47

    vsize = 47#128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    log_interval = 100
    save_interval = 50

    max_step = 100000

    n_workers = 0


class DQNRNNHyperParams:
    vision = 'VAE'
    memory = 'RNN'
    n_hiddens = 256
    extra = False
    # data_dir = 'dqnDatasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'
    seed = 0    

    

    batch_size = 64 # actually batchsize * Seqlen
    test_batch = 1
    seq_len = 10
    n_sample = 47

    vsize = 47#128 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size
    rnn_hunits = 256
    log_interval = 100
    save_interval = 50

    max_step = 100000

    n_workers = 0



class VAEHyperParams:
    vision = 'VAE'

    extra = False
    data_dir = 'Datasets'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0


class VAEHyperParamsTimestep05:
    vision = 'VAE'

    extra = False
    data_dir = 'RobotFrameDatasetsTimestep05'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0



# class VAEHyperParamsTimestepcontinuous:
#     vision = 'VAE'

#     extra = False
#     # data_dir = 'RobotFrameContinuousDatasetsTimestep1'
#     data_dir = 'RobotFrameContinuousDatasetsTimestep1'
#     extra_dir = 'additional'
#     ckpt_dir = 'ckpt'

#     n_hiddens = 256
#     batch_size = 64 # 
#     test_batch = 12
#     n_sample = 64

#     vsize = 47 # latent size of Vision
#     msize = 128 # size of Memory
#     asize = 2 # action size

#     log_interval = 5000
#     save_interval = 10000

#     max_step = 100_0000

#     n_workers = 0

class VAEHyperParamsTimestepcontinuous:
    vision = 'VAE'

    extra = False
    # data_dir = 'RobotFrameContinuousDatasetsTimestep1'
    data_dir = 'RobotFrameContinuousDatasetsTimestep1c'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0


class VAEHyperParamsTimestepcontinuous_1:
    vision = 'VAE'

    extra = False
    data_dir = 'RobotFrameContinuousDatasetsTimestep1'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0


class VAEHyperParamsTimestepcontinuous025:
    vision = 'VAE'

    extra = False
    # data_dir = 'RobotFrameContinuousDatasetsTimestep1'
    data_dir = 'RobotFrameContinuousDatasetsTimestep025'
    extra_dir = 'additional'
    ckpt_dir = 'ckpt'

    n_hiddens = 256
    batch_size = 64 # 
    test_batch = 12
    n_sample = 64

    vsize = 47 # latent size of Vision
    msize = 128 # size of Memory
    asize = 2 # action size

    log_interval = 5000
    save_interval = 10000

    max_step = 100_0000

    n_workers = 0