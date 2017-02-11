# NIST-Traffic-Data-Prediction-Flow

Contains detailed problem statement in NIST-TrafficDataPrediction.pdf and detailed solution in FlowPredictionReport.pdf

Problem Statement: This was part of NIST traffic data prediction and cleaning task. I was a given a collection of erroneous traffic flow, speed and occupancy data and I was asked to predict the correct flow measurement. The measurements had probabilities associated with them.
Sol: I calculated predicted values as a weighted average of the following:
a) Linear Regression with traffic flow from nearby lanes
b) Weighted average of traffic flow values from neighboring timestamps
c) Original flow value
