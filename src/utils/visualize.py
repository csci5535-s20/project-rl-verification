import numpy as np
import matplotlib
matplotlib.use('Agg')
from statistics import mean
from matplotlib import animation
import matplotlib.pyplot as plt
from collections import deque
import os
import csv


AVERAGE_SCORE_TO_SOLVE = 195 # fixed threshold
CONSECUTIVE_RUNS_TO_SOLVE = 100


class VisualizeScores:

  def __init__(self, env_name):
    self.scores = deque(maxlen=CONSECUTIVE_RUNS_TO_SOLVE)
    self.env_name = env_name

  def add_score(self, score, run, output_path):
    CSV_PATH = output_path + '.csv'
    PNG_PATH = output_path + '.png'

    self.save_to_csv(CSV_PATH, score)
    self.save_to_png(saved_csv_path=CSV_PATH,
                     output_path=PNG_PATH,
                     x_label='runs',
                     y_label='scores',
                     average_of_n_last=CONSECUTIVE_RUNS_TO_SOLVE,
                     show_goal=True,
                     show_trend=True,
                     show_legend=True)

    self.scores.append(score)
    mean_score = mean(self.scores)
    print('Scores: (min: {}, avg: {}, max: {})\n'.format(min(self.scores), np.round(mean_score,2), max(self.scores)))

    # scenario is solved when mean >= fixed avg score and number of episodes > fxied runs to solve
    if mean_score >= AVERAGE_SCORE_TO_SOLVE and len(self.scores) >= CONSECUTIVE_RUNS_TO_SOLVE:
      solve_score = run - CONSECUTIVE_RUNS_TO_SOLVE
      print('Solved in {} runs  and {} total runs'.format(solve_score, run))

      self.save_to_csv('solved.csv', solve_score)
      self.save_to_png(saved_csv_path='solved.csv',
                       output_path='solved.png',
                       x_label='Trials',
                       y_label='Steps',
                       average_of_n_last=None,
                       show_goal=False,
                       show_trend=False,
                       show_legend=False)
      print('Exiting ...\n')
      exit()

  def save_to_png(self, saved_csv_path, output_path, x_label, y_label, average_of_n_last, show_goal, show_trend, show_legend):
    x = []
    y = []
    with open(saved_csv_path, 'r') as scores:
      reader = csv.reader(scores)
      data = list(reader)
      for i in range(0, len(data)):
        x.append(int(i))
        y.append(int(data[i][0]))

    plt.subplots()
    plt.plot(x, y, label='score per run')

    average_range = average_of_n_last if average_of_n_last is not None else len(x)
    plt.plot(x[-average_range:], [np.mean(y[-average_range:])] * len(y[-average_range:]), linestyle='--', \
             label='Average of last {} runs'.format(average_range))

    if show_goal:
      plt.plot(x, [AVERAGE_SCORE_TO_SOLVE] * len(x), linestyle=':', \
               label='Average solving score: {}'.format(AVERAGE_SCORE_TO_SOLVE))

    if show_trend and len(x) > 1:
      trend_x = x[1:]
      z = np.polyfit(np.array(trend_x), np.array(y[1:]), 1)
      p = np.poly1d(z)
      plt.plot(trend_x, p(trend_x), linestyle="-.",  label='trend')

    plt.title(self.env_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if show_legend:
      plt.legend(loc="upper left")

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

  def save_to_csv(self, path, score):
    if not os.path.exists(path):
      with open(path, "w"):
        pass
    scores_file = open(path, "a")
    with scores_file:
      writer = csv.writer(scores_file)
      writer.writerow([score])


  def save_frames_as_gif(frames, path='./', filename='cartpole_animation.gif'):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')
    def animate(i):
      patch.set_data(frames[i])
    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    plt.close()
