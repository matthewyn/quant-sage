import re
import math
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
from tqdm.notebook import tqdm
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import confusion_matrix

GREEN = "\033[92m"
RED   = "\033[91m"
RESET = "\033[0m"

WORKERS      = 3
DEFAULT_SIZE = 200


class Tester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE, workers=WORKERS):
        self.predictor   = predictor
        self.data        = data
        self.title       = title or self.make_title(predictor)
        self.size        = size
        self.workers     = workers
        self.titles      = []
        self.predictions = []   # "UP" or "DOWN"
        self.actuals     = []   # "UP" or "DOWN"
        self.corrects    = []   # bool per datapoint
        self.correct     = 0

    @staticmethod
    def make_title(predictor) -> str:
        return predictor.__name__.replace("__", ".").replace("_", " ").title().replace("Gpt", "GPT")

    @staticmethod
    def parse_direction(value) -> str:
        """Extract UP or DOWN from model output. Defaults to UP if unclear."""
        if isinstance(value, str):
            upper = value.upper()
            if "DOWN" in upper:
                return "DOWN"
            if "UP" in upper:
                return "UP"
        return "UP"  # fallback

    def run_datapoint(self, i):
        datapoint = self.data[i]
        raw       = self.predictor(datapoint)
        predicted = self.parse_direction(raw)
        actual    = str(datapoint["completion"]).strip().upper()  # "UP" or "DOWN"
        correct   = predicted == actual
        pieces = datapoint["prompt"].split("TREND: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."
        return title, predicted, actual, correct

    def running_accuracy_chart(self):
        n   = len(self.corrects)
        x   = list(range(1, n + 1))

        running_sums = list(accumulate(int(c) for c in self.corrects))
        running_acc  = [s / i * 100 for s, i in zip(running_sums, x)]

        # Wilson 95% CI for a proportion
        ci    = [1.96 * math.sqrt((a / 100) * (1 - a / 100) / i) * 100
                 for a, i in zip(running_acc, x)]
        upper = [min(a + c, 100) for a, c in zip(running_acc, ci)]
        lower = [max(a - c,   0) for a, c in zip(running_acc, ci)]

        final_acc = running_acc[-1]
        final_ci  = ci[-1]

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x + x[::-1], y=upper + lower[::-1],
            fill="toself", fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip", showlegend=False,
        ))

        fig.add_trace(go.Scatter(
            x=x, y=running_acc,
            mode="lines", line=dict(width=3, color="steelblue"),
            name="Running Accuracy",
            customdata=ci,
            hovertemplate="n=%{x}<br>Accuracy=%{y:.1f}%<br>±95CI=%{customdata:.1f}%<extra></extra>",
        ))

        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="50% (random)", annotation_position="bottom right")

        fig.update_layout(
            title=f"{self.title} — Running Accuracy: {final_acc:.1f}% ± {final_ci:.1f}%",
            xaxis_title="Number of Datapoints",
            yaxis_title="Directional Accuracy (%)",
            yaxis=dict(range=[30, 80]),
            width=1000, height=360,
            template="plotly_white", showlegend=False,
        )
        fig.show()

    def confusion_matrix_chart(self):
        cm     = confusion_matrix(self.actuals, self.predictions, labels=["UP", "DOWN"])
        labels = ["UP", "DOWN"]
        z_text = [[str(v) for v in row] for row in cm.tolist()]

        fig = go.Figure(go.Heatmap(
            z=cm,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}"    for l in labels],
            text=z_text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
        ))
        fig.update_layout(
            title=f"{self.title} — Confusion Matrix",
            width=450, height=400,
        )
        fig.show()

    def accuracy_by_class_chart(self):
        df = pd.DataFrame({"actual": self.actuals, "predicted": self.predictions})
        df["correct"] = df["actual"] == df["predicted"]

        by_class = (df.groupby("actual")["correct"]
                      .agg(["sum", "count"])
                      .assign(accuracy=lambda d: d["sum"] / d["count"] * 100)
                      .reset_index())

        fig = px.bar(
            by_class, x="actual", y="accuracy",
            color="actual",
            color_discrete_map={"UP": "green", "DOWN": "red"},
            text=by_class["accuracy"].apply(lambda v: f"{v:.1f}%"),
            labels={"actual": "Actual Direction", "accuracy": "Accuracy (%)"},
            title=f"{self.title} — Accuracy by Class",
            width=450, height=400,
        )
        fig.add_hline(y=50, line_dash="dash", line_color="gray")
        fig.update_traces(textposition="outside")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(showlegend=False)
        fig.show()

    def report(self):
        directional_accuracy = self.correct / self.size * 100

        df = pd.DataFrame({"actual": self.actuals, "predicted": self.predictions})
        df["correct"] = df["actual"] == df["predicted"]

        up_acc   = df[df["actual"] == "UP"]["correct"].mean()   * 100
        down_acc = df[df["actual"] == "DOWN"]["correct"].mean() * 100
        up_pct   = (df["predicted"] == "UP").mean() * 100

        print(f"\n{'='*50}")
        print(f"  {self.title}")
        print(f"{'='*50}")
        print(f"  Directional Accuracy : {directional_accuracy:.1f}%")
        print(f"  UP   Accuracy        : {up_acc:.1f}%")
        print(f"  DOWN Accuracy        : {down_acc:.1f}%")
        print(f"  % Predicted UP       : {up_pct:.1f}%")
        print(f"  Total evaluated      : {self.size}")
        print(f"{'='*50}\n")

        self.running_accuracy_chart()
        self.confusion_matrix_chart()
        self.accuracy_by_class_chart()

    def run(self):
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            for title, predicted, actual, correct in tqdm(
                ex.map(self.run_datapoint, range(self.size)), total=self.size
            ):
                self.titles.append(title)
                self.predictions.append(predicted)
                self.actuals.append(actual)
                self.corrects.append(correct)
                if correct:
                    self.correct += 1
                color_code = GREEN if correct else RED
                print(f"{color_code}{'✓' if correct else '✗'}{RESET} ", end="")
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE, workers=WORKERS):
    Tester(function, data, size=size, workers=workers).run()