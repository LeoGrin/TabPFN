import openml

suites = [337, 334]

for suite_id in suites:
    suite = openml.study.get_suite(suite_id)
    tasks = suite.tasks
    for task_id in tasks:
        task = openml.tasks.get_task(task_id) 
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
        dataset_format="dataframe", target=dataset.default_target_attribute
    )
