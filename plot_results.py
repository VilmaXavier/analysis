# plot_results.py
import matplotlib.pyplot as plt

def plot_comparison(metrics_dict):
    models = list(metrics_dict.keys())
    metrics = list(metrics_dict[models[0]].keys())

    for metric in metrics:
        scores = [metrics_dict[model][metric] for model in models]
        plt.bar(models, scores)
        plt.title(metric)
        plt.ylabel("Score")
        plt.ylim(0, 1)
        plt.show()
