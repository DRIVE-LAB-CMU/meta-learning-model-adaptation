
DatasetConfig = dict(
    real_dataset_1 = dict(
        train_data_list = [
            dict(name='real-random', dir='data-20240130-230806', range=[10, 156]),
            dict(name='real-random', dir='data-20240130-231331', range=[10, 124]),
            dict(name='real-random', dir='data-20240130-231613', range=[10, 198]),
            dict(name='real-random', dir='data-20240130-231924', range=[10, 202]),
            dict(name='real-random', dir='data-20240130-232244', range=[10, 525]),
            dict(name='real-random', dir='data-20240130-232853', range=[10, 321]),
        ],
        test_data_list = [ ]
    ),
    sim_dataset_1 = dict(
        train_data_list = [
            dict(name='sim-random', dir='sim-data-20240131-134617', range=[0, 1]),
        ],
        test_data_list = [ ]
    ),
    sim_dataset_2 = dict(
        train_data_list = [
            dict(name='sim-random', dir='sim-data-20240201-190458', range=[0, 1]),
        ],
        test_data_list = [ ]
    ),
    mix_dataset_1 = dict(
        train_data_list = [
            dict(name='real-random', dir='data-20240130-230806', range=[10, 156]),
            dict(name='real-random', dir='data-20240130-231331', range=[10, 124]),
            dict(name='real-random', dir='data-20240130-231613', range=[10, 198]),
            dict(name='real-random', dir='data-20240130-231924', range=[10, 202]),
            dict(name='real-random', dir='data-20240130-232244', range=[10, 525]),
            dict(name='real-random', dir='data-20240130-232853', range=[10, 321]),
            dict(name='sim-random', dir='sim-data-20240131-134617', range=[0, 1]),
        ],
        test_data_list = [ ]
    ),
    real_dataset_2 = dict(
        train_data_list = [
            dict(name='real-random', dir='data-20240201-200427', range=[5, 480]),
            dict(name='real-random', dir='data-20240201-202035', range=[5, 489]),
            dict(name='real-random', dir='data-20240201-221923', range=[5, 689]),
            dict(name='real-random', dir='data-20240130-230806', range=[10, 156]),
            dict(name='real-random', dir='data-20240130-231331', range=[10, 124]),
            dict(name='real-random', dir='data-20240130-231613', range=[10, 198]),
            dict(name='real-random', dir='data-20240130-231924', range=[10, 202]),
            dict(name='real-random', dir='data-20240130-232244', range=[10, 525]),
            dict(name='real-random', dir='data-20240130-232853', range=[10, 321]),
            dict(name='real-random', dir='data-20240131-184333', range=[10, 91]),
        ],
        test_data_list = [ ]
    ),
    mix_dataset_2 = dict(
        train_data_list = [
            dict(name='real-random', dir='data-20240201-200427', range=[5, 480]),
            dict(name='real-random', dir='data-20240201-202035', range=[5, 489]),
            dict(name='real-random', dir='data-20240201-221923', range=[5, 689]),
            dict(name='sim-random', dir='sim-data-20240131-134617', range=[0, 1]),
        ],
        test_data_list = [ ]
    ),
    
    real_dataset_3 = dict(
        train_data_list = [
            dict(name='real-circle', dir='vicon-circle-data-20240304-172815-1.csv'),
            dict(name='real-circle', dir='vicon-circle-data-20240304-161941-1.csv'),
        ],
        test_data_list = [ ]
    )
            
)