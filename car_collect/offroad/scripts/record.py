
import imageio.v2 as imageio
import matplotlib
import os
from rich.progress import track
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('logdir', 'data-20231219-122003', 'log dir.')

def main(argv):
    log_dir = FLAGS.logdir

    if not os.path.exists(os.path.join('../video', log_dir)):
        os.makedirs(os.path.join('../video', log_dir))

    image_list = []
    for i in track(range(10000)): #480
        try:
            fig, axs = plt.subplots(2, 2, figsize=(10, 6))
            plt.subplots_adjust(wspace=0, hspace=0)
            for row in range(2):
                for col in range(2):
                    axs[row, col].axis('off')  # Removes axes and ticks
            axs[0][0].imshow(imageio.imread(f"tmp/{log_dir}/img_left_{i}.png"))
            axs[0][1].imshow(imageio.imread(f"tmp/{log_dir}/img_right_{i}.png"))
            axs[1][0].imshow(imageio.imread(f"tmp/{log_dir}/lidar_{i}.png"))
            axs[1][1].imshow(imageio.imread(f"tmp/{log_dir}/loc_{i}.png"))
            plt.savefig(f"tmp/{log_dir}/summary_{i}.png",bbox_inches='tight', pad_inches=0)
            image_list.append(imageio.imread(f"tmp/{log_dir}/summary_{i}.png"))
            plt.close()
        except:
            break

    imageio.mimsave(f"../video/{log_dir}/summary.gif", image_list, duration=len(image_list)*0.01)
    
if __name__ == '__main__':
    app.run(main)