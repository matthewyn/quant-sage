import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from itertools import accumulate
import math
from tqdm.auto import tqdm
from IPython.display import clear_output
from sklearn.metrics import confusion_matrix

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"

DEFAULT_SIZE = 200


class Tester:
    def __init__(self, predictor, data, title=None, size=DEFAULT_SIZE):
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = size
        self.titles = []
        self.predictions = []   # "UP" or "DOWN"
        self.actuals = []       # "UP" or "DOWN"
        self.corrects = []      # True/False per datapoint
        self.correct = 0

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
        raw = self.predictor(datapoint)
        predicted = self.parse_direction(raw)
        actual = datapoint["completion"].strip().upper()  # "UP" or "DOWN"

        correct = predicted == actual

        pieces = datapoint["prompt"].split("TREND: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."

        color = "green" if correct else "red"
        return title, predicted, actual, correct, color

    def running_accuracy_chart(self):
        n = len(self.corrects)
        x = list(range(1, n + 1))

        running_sums = list(accumulate(int(c) for c in self.corrects))
        running_acc = [s / i * 100 for s, i in zip(running_sums, x)]

        # 95% Wilson confidence interval for proportion
        ci = [
            1.96 * math.sqrt((acc / 100) * (1 - acc / 100) / i) * 100
            for acc, i in zip(running_acc, x)
        ]
        upper = [min(a + c, 100) for a, c in zip(running_acc, ci)]
        lower = [max(a - c, 0)   for a, c in zip(running_acc, ci)]

        final_acc = running_acc[-1]
        final_ci  = ci[-1]
        title = f"{self.title} — Running Accuracy: {final_acc:.1f}% ± {final_ci:.1f}%"

        fig = go.Figure()

        # Shaded CI band
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(128,128,128,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=False,
        ))

        # Running accuracy line
        fig.add_trace(go.Scatter(
            x=x,
            y=running_acc,
            mode="lines",
            line=dict(width=3, color="steelblue"),
            name="Running Accuracy",
            customdata=ci,
            hovertemplate="n=%{x}<br>Accuracy=%{y:.1f}%<br>±95CI=%{customdata:.1f}%<extra></extra>",
        ))

        # 50% baseline
        fig.add_hline(y=50, line_dash="dash", line_color="red",
                      annotation_text="50% (random)", annotation_position="bottom right")

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Directional Accuracy (%)",
            yaxis=dict(range=[30, 80]),
            width=800, height=350,
            template="plotly_white",
            showlegend=False,
        )
        fig.show()

    def confusion_matrix_chart(self):
        cm = confusion_matrix(self.actuals, self.predictions, labels=["UP", "DOWN"])
        # cm[i][j] = count where actual=i, predicted=j
        labels = ["UP", "DOWN"]
        z = cm.tolist()
        z_text = [[str(v) for v in row] for row in z]

        fig = go.Figure(go.Heatmap(
            z=z,
            x=[f"Predicted {l}" for l in labels],
            y=[f"Actual {l}"    for l in labels],
            text=z_text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        ))
        fig.update_layout(
            title=f"{self.title} — Confusion Matrix",
            width=450, height=400,
        )
        fig.show()

    def accuracy_by_class_chart(self):
        df = pd.DataFrame({"actual": self.actuals, "predicted": self.predictions})
        df["correct"] = df["actual"] == df["predicted"]

        by_class = df.groupby("actual")["correct"].agg(["sum", "count"])
        by_class["accuracy"] = by_class["sum"] / by_class["count"] * 100
        by_class = by_class.reset_index()

        fig = px.bar(
            by_class,
            x="actual", y="accuracy",
            color="actual",
            color_discrete_map={"UP": "green", "DOWN": "red"},
            text=by_class["accuracy"].apply(lambda x: f"{x:.1f}%"),
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

        up_acc   = df[df["actual"] == "UP"]["correct"].mean() * 100
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
        for i in tqdm(range(self.size)):
            title, predicted, actual, correct, color = self.run_datapoint(i)
            self.titles.append(title)
            self.predictions.append(predicted)
            self.actuals.append(actual)
            self.corrects.append(correct)
            if correct:
                self.correct += 1
            color_code = GREEN if correct else RED
            print(f"{color_code}{'✓' if correct else '✗'}{RESET} ", end="")
        clear_output(wait=True)
        self.report()


def evaluate(function, data, size=DEFAULT_SIZE):
    Tester(function, data, size=size).run()