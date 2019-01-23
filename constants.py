

grid_size_x = 4
grid_size_y = 4

when_to_save_dict = {1:15, 3: 15, 9:15, 15: 30, 150: 50, 1500: 100} # num_schedules : when_to_save
epoch_range = 5000
epoch_range_dict = {3:2000, 9: 1500, 15: 1000, 150: 500, 1500: 50}
loss_multiplier = 30
loss_multiplier_dict = {3:25, 9:20, 15:20, 150:25}
rollout = 4


lr_network = .0001
lr_embedding = .001
noise_percentage = 0
epoch_save_num = epoch_range/10

test_loss_multiplier = 20








######################## PAIRWISE ###########################3
loss_multiplier_pairwise = 1