import matplotlib.pyplot as plt
import numpy as np
from useful import remove_parts_of_graph_encoder_contiformer
from create_data import callFunction
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--y_lim_low', type=int, default=10)
parser.add_argument('--y_lim_high', type=int, default=10000)
parser.add_argument('--train_count', type=int, default=10)
parser.add_argument('--val_count', type=int, default=1)
parser.add_argument('--number_x_values', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=10) #ausprobieren
parser.add_argument('--random_number_range_distribution', type=str, default="norm")
parser.add_argument('--random_number_range_mean', type=int, default=0)
parser.add_argument('--random_number_range_std', type=int, default=5)
parser.add_argument('--spline_value_low', type=int, default=800000)
parser.add_argument('--spline_value_high', type=int, default=1100000)
parser.add_argument('--vocab_size', type=int, default=100000)
parser.add_argument('--noise_std_distribution', type=str, default="norm")
parser.add_argument('--noise_std_mean', type=int, default=0)
parser.add_argument('--noise_std_std', type=int, default=0.15)
parser.add_argument('--interpolation_min_width', type=int, default=10)
parser.add_argument('--interpolation_max_width', type=int, default=100)
parser.add_argument('--interpolation_max_count', type=int, default=10)
parser.add_argument('--offset', type=int, default=10)
parser.add_argument('--x_lim_low', type=int, default=0)
parser.add_argument('--x_lim_high', type=int, default=1000)

args = parser.parse_args()

if __name__ == "__main__":

    for i in range(100):
        y_start = np.random.uniform(10 + 1,10000 - 1)

        timeSeries = [0,1,2,3,4,5,6,7]
        randomInt = np.random.choice(timeSeries)
        randomInt = 3

        x_values = np.arange(1000)

        y_spline, y_noise_spline,min_value, max_value, noise_std = callFunction(x_values=x_values, y_start=y_start, random_number_range=[args.random_number_range_distribution, args.random_number_range_mean, args.random_number_range_std], spline_value=[args.spline_value_low, args.spline_value_high], vocab_size=args.vocab_size, randomInt=randomInt, noise_std=[args.noise_std_distribution, args.noise_std_mean, args.noise_std_std])


        mask = remove_parts_of_graph_encoder_contiformer(x_values, 500, 10)
        # y_values[mask == 1] = 0
        print(np.sum(mask == 1))

        fig, ax = plt.subplots(1,1)
        y_noise_spline[mask == 1] = np.nan
        ax.plot(x_values, y_noise_spline)
        ax.plot(x_values, y_spline)
        plt.show(block=False)
        plt.pause(3.00)       
        plt.close()