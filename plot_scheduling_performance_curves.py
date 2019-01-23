import matplotlib.pyplot as plt
import numpy as np

def main():
    top_1_mean_bins_NNs = [5.96, 7.55, 7.98, 12.44]
    std_1_mean_bins_NNs = [5.0, 5.58, 4.88, 7.7]
    top_3_mean_bins_NNs = [15.45, 18.90, 19.62, 29.71]
    std_3_mean_bins_NNs = [7.33, 9.05, 8.30, 9.9]

    top_1_mean_bins_kNNs = [5.32, 5.94, 5.86, 11.05]
    std_1_mean_bins_kNNs = [4.86, 7.39, 5.8, 5.5]
    top_3_mean_bins_kNNs = [5.32, 18.36, 18.39, 27.46]
    std_3_mean_bins_kNNs = [4.86, 7.46, 9.2, 8.1]

    top_1_mean_bins_BNNs = [7.04, 6.03, 7.95, 69.44]
    std_1_mean_bins_BNNs = [5.31, 5.58, 5.6, 11.55]
    top_3_mean_bins_BNNs = [14.83, 19.11, 21.42, 89.2]
    std_3_mean_bins_BNNs = [7.57, 9.05, 7.8, 9.5]

    top_1_mean_bins_gNNs = [4.72, 7.07, 5.77, 11.34]
    std_1_mean_bins_gNNs = [4.47, 5.77, 4.8, 6.45]
    top_3_mean_bins_gNNs = [14.60, 19.11, 18.16, 26.61]
    std_3_mean_bins_gNNs = [7.8, 8.90, 8.4, 8.29]

    top_1_mean_bins_lstm = [4.85, 6.26, 5.28, 4.45]
    std_1_mean_bins_lstm = [4.53, 5.19, 4.0, 1.6]
    top_3_mean_bins_lstm = [15.45, 16.91, 15.84, 14.4]
    std_3_mean_bins_lstm = [7.17, 8.9, 7.98, 2.04]

    top_1_mean_bins_blstm = [4.46, 5.00, 5.93, 4.57]
    std_1_mean_bins_blstm = [4.13, 3.85, 3.7, 1.8]
    top_3_mean_bins_blstm = [14.47, 17.02, 17.7, 14.6]
    std_3_mean_bins_blstm = [6.2, 7.7, 7.5, 4.9]

    num_scheds = [3, 9, 15, 150]
    labels = ['Baseline NN', 'k-means -> NN', 'GMM -> NN', 'BNN', 'LSTM', 'B-LSTM']
    plt.errorbar(num_scheds, top_1_mean_bins_NNs, [x / np.sqrt(50) for x in std_1_mean_bins_NNs], label=labels[0],  
           linestyle='-'
             , capsize=5)
    plt.errorbar(num_scheds, top_1_mean_bins_kNNs, [x / np.sqrt(50) for x in std_1_mean_bins_kNNs], label=labels[1],
           linestyle='-'
             , capsize=5)
    plt.errorbar(num_scheds, top_1_mean_bins_gNNs, [x / np.sqrt(50) for x in std_1_mean_bins_gNNs], label=labels[2],  
           linestyle='-'
             , capsize=5)
    plt.errorbar(num_scheds, top_1_mean_bins_BNNs, [x / np.sqrt(50) for x in std_1_mean_bins_BNNs], label=labels[3],  
           linestyle='-'
             , capsize=5)

    plt.errorbar(num_scheds, top_1_mean_bins_lstm, [x / np.sqrt(50) for x in std_1_mean_bins_lstm], label=labels[4],  
           linestyle='-'
             , capsize=5)
    plt.errorbar(num_scheds, top_1_mean_bins_blstm, [x / np.sqrt(50) for x in std_1_mean_bins_blstm], label=labels[5],  
          linestyle='-'
             , capsize=5)
    plt.ylabel('Task Prediction Accuracy (%)')
    plt.xscale('log')
    plt.xlabel('Number of Schedules Trained Upon')
    plt.legend()
    plt.xlim(2,160)
    plt.ylim(2,100)
    plt.show()


    plt.errorbar(num_scheds, top_3_mean_bins_NNs, [x / np.sqrt(50) for x in std_3_mean_bins_NNs],  label=labels[0],
          linestyle='-'
             , capsize=5)
    plt.errorbar(num_scheds, top_3_mean_bins_kNNs, [x / np.sqrt(50) for x in std_3_mean_bins_kNNs], label=labels[1],
                 linestyle='-', capsize=5
                 )
    plt.errorbar(num_scheds, top_3_mean_bins_gNNs, [x / np.sqrt(50) for x in std_3_mean_bins_gNNs], label=labels[2],
                 linestyle='-', capsize=5
                 )
    plt.errorbar(num_scheds, top_3_mean_bins_BNNs, [x / np.sqrt(50) for x in std_3_mean_bins_BNNs], label=labels[3],
                 linestyle='-', capsize=5
                 )

    plt.errorbar(num_scheds, top_3_mean_bins_lstm, [x / np.sqrt(50) for x in std_3_mean_bins_lstm], label=labels[4],  
          linestyle='-', capsize=5)
    plt.errorbar(num_scheds, top_3_mean_bins_blstm, [x / np.sqrt(50) for x in std_3_mean_bins_blstm], label=labels[5],
          linestyle='-', capsize=5)
    plt.ylabel('Task Prediction Accuracy (%)')
    plt.xscale('log')
    plt.xlabel('Number of Schedules Trained Upon')
    plt.legend()
    plt.xlim(2, 160)
    plt.ylim(1, 100)
    plt.show()


if __name__ == '__main__':
    main()
