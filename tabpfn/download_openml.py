from sklearn.datasets import fetch_openml

# suites = [337, 334]

# for suite_id in suites:
#     print("suite_id", suite_id)
#     suite = openml.study.get_suite(suite_id)
#     tasks = suite.tasks
#     for task_id in tasks:
#         task = openml.tasks.get_task(task_id) 
#         dataset = task.get_dataset()
#         print(dataset.id)

dataset_ids = [44089, 44120, 44121, 44122, 44123, 44125, 44126, 44128, 44129, 44130, 45022,
       45021, 45020, 45019, 45028, 45026, 44156, 44157, 44159, 45035, 45036, 45038, 45039]

for dataset_id in dataset_ids:
    # download the dataset
    data = fetch_openml(data_id=dataset_id, as_frame=True)
