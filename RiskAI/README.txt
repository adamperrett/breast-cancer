This one is a little confusing and very messy, my main project, good luck. Objective is not to calculate VAS, but to predict cancer outcomes.

fold_scheduler.py is ran on the csf to schedule each fold as a separate x-val fold (cheeky way to bypass timelimits) and calls X jobs with script run_fold.py
run_fold.py is the main train script

