import workflow
import numpy as np
import scipy.ndimage
import skimage.measure
import optuna
import datasets as ds
import diagnostics
import sklearn.preprocessing
import sklearn.feature_selection
import sklearn.model_selection
import sklearn.linear_model
from pieces import slack


def OptunaRun(project_name, run_name, main_path, folders, 
              bin_spatial=4, bin_spectral=20, slice_spatial=slice(40, 280), slice_spectral=slice(0, 1200), 
              N_size=-1, test_size=0.2, n_trials=400, debug_mode=False):
    global study, best_accuracy

    def new_score(x, y):
        scores, b = sklearn.feature_selection.f_classif(x, y)
        scores = np.log(scores+0.1)
        scores += np.log(ps_max)
        return scores, b
        
    def objective(trial):   
        global best_accuracy
        selector.k = trial.suggest_int("k_2", 500, all_values.shape[1])
        selected_features = selector.transform(all_values)
    
        (x_train, x_test, y_train, y_test,
        i_train, i_test) = sklearn.model_selection.train_test_split(
            selected_features, all_labels, indices,
            test_size=test_size, shuffle=True, stratify=all_labels
        )
    
        classifier = sklearn.linear_model.LogisticRegression(
            C=trial.suggest_float('C', 1e-3, 1e3, log=True),
            class_weight='balanced',
            max_iter=8000
        )
        classifier.fit(x_train, y_train)
    
        predictions = classifier.predict(x_test)
        accuracy = sklearn.metrics.balanced_accuracy_score(y_test, predictions)
    
        # Save assorted data
        best = accuracy > best_accuracy
        cm = diagnostics.confusion_matrix(
            flow, trial.number, accuracy,
            y_test, predictions, best
        )
        if best:
            best_accuracy = accuracy
            selected_indices = selector.get_support(indices=True)
            flow.save_metadata(classifier.coef_, "best_weights")
            flow.save_metadata(selected_indices, "best_feature_indices")
            flow.save_metadata(i_test, "best_test_indices")
            flow.save_metadata(cm, "best_confusion_matrix")
            diagnostics.plot_weights(
                flow, classifier.coef_,
                selected_indices, original_shape, len(folders)
            )
            diagnostics.plot_scores_features(
                flow, scores, original_shape,
                selected_indices,
            )
        return accuracy

    def run_study():
        global best_accuracy, study
        flow.status("Running Optuna study")
        best_accuracy = 0
        study = optuna.create_study(direction='maximize')
        try:
            study.optimize(
                objective, n_trials=150, show_progress_bar=True,
                callbacks=(lambda s, t: diagnostics.study_history(flow, s),),
            )
        except KeyboardInterrupt:
            flow.status("Optuna study interrupted")
        flow.status("Study done, saving")
        flow.save_metadata(study, "study")
        flow.status(f"All done, best acc: {study.best_value*100:.5f}%")
        
    # Create workflow
    Flow = workflow.SlackFlow
    # Flow = workflow.Flow
    if debug_mode:
        flow = Flow(f'[DEBUG]{project_name}')
    else:
        flow = Flow(project_name)
    flow.status(f"Setting up - {run_name}")
    flow.set_context(run_name)
    
    optuna.progress_bar.tqdm = flow.tqdm

    # Load files
    flow.status("Loading files")
    all_values = []
    ps_maxes = []
    
    for folder in folders:
        flow.status(f"Loading folder: {folder}")
        fnames = ds.glob(f"{main_path}/{folder}/*.ds", no=("meta",))
        folder_values = []
        all_labels = []
    
        if debug_mode:
            N = 1
            flow.status('test_size is overrided to 0.2 in debug mode')
            test_size = 0.2
        else:
            N = len(fnames)
        
        for fname in flow.tqdm(fnames[:N]):
            # Load data in
            data = ds.load(fname)
            meta = ds.load(fname.replace(".ds", "_meta.ds"))
        
            # Subtract background
            bg_step = 10
            K = np.ones([bg_step, bg_step])/(bg_step * bg_step);
            background = scipy.ndimage.convolve(meta["background"].astype('float32'), K, mode='constant')
            values = data.raw.astype('float32') - background
    
            values = skimage.measure.block_reduce(values[:, slice_spatial, slice_spectral], (1, bin_spatial, bin_spectral))
            original_shape = values.shape[1:]
            values = values.reshape((values.shape[0], -1))
        
            # Scale
            ps_max = np.amax(values, 0)
            scaler = sklearn.preprocessing.RobustScaler()
            values = scaler.fit_transform(values)
        
            # Save in an array
            folder_values.append(values)
            all_labels.extend(data.label)

            if len(folder_values)*folder_values[0].shape[0] >= N_size and N_size != -1:
                folder_values = np.concatenate(folder_values)
                folder_values = folder_values[slice(None, N_size)]
                all_labels = all_labels[slice(None, N_size)]
                flow.status(f"Early termination of data extraction. The first {N_size} data is extracted")
                break
        if N_size == -1 or debug_mode:
            folder_values = np.concatenate(folder_values)
        all_values.append(folder_values)
        ps_maxes.append(ps_max)
    flow.status("Merging folders")
    all_values = np.concatenate(all_values, 1)
    ps_max = np.concatenate(ps_maxes)
    
    print(all_values.min())
    
    diagnostics.plot_maxima(flow, all_values, original_shape)
    diagnostics.plot_minima(flow, all_values, original_shape)


    # if not debug_mode:
    scores = diagnostics.plot_scores(flow, all_values, all_labels, original_shape, function=new_score)
    
    selector = sklearn.feature_selection.SelectKBest(k=0, score_func=new_score)
    selector.fit(all_values, all_labels)
    
    indices = np.arange(all_values.shape[0])
    
    flow.status("Running Optuna study")
    best_accuracy = 0
    study = optuna.create_study(direction='maximize')
    try:
        slack.send('---')
        study.optimize(
            objective, n_trials=n_trials, show_progress_bar=True,
            callbacks=(lambda s, t: diagnostics.study_history(flow, s),),
        )
    except KeyboardInterrupt:
        flow.status("Optuna study interrupted")
    
    flow.status("Study done, saving")
    flow.save_metadata(study, "study")
    flow.status(f"All done, best acc: {study.best_value*100:.5f}%")
    
    flow.close()
