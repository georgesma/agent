class TrainingRecord:
    def __init__(self):
        self.record = {}

    def save_epoch_metrics(self, epoch_step, epoch_metrics):
        if epoch_step not in self.record:
            self.record[epoch_step] = {}
        for metric_name, metric_value in epoch_metrics.metrics.items():
            if metric_name not in self.record[epoch_step]:
                self.record[epoch_step][metric_name] = []
            self.record[epoch_step][metric_name].append(metric_value)

        self.log_metrics(epoch_step, epoch_metrics)

    def log_metrics(self, epoch_step, epoch_metrics):
        print(epoch_step)
        longest_name_len = max([len(name) for name in epoch_metrics.metrics])
        for metric_name, metric_value in epoch_metrics.metrics.items():
            print(f"- {metric_name: <{longest_name_len}} | {metric_value:.4f}")


class EpochMetrics:
    def __init__(self, nb_batch):
        self.metrics = {}
        self.nb_batch = nb_batch

    def add(self, metric_name, metric_value):
        if metric_name not in self.metrics:
            self.metrics[metric_name] = 0
        self.metrics[metric_name] += metric_value / self.nb_batch
