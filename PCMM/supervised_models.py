# Minimal, well-commented helper functions for baseline supervised models
# Dependencies: numpy, sklearn
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA

def _make_groups_from_counts(counts):
    """
    Make a group label array from counts-per-subject.
    counts : iterable of ints (length = n_subjects)
    returns groups : ndarray length = sum(counts)
    """
    if counts is None:
        return None
    elif isinstance(counts, np.int64):
        #repeat counts
        counts = 155*[counts]
    counts = np.asarray(counts, dtype=int)
    if counts.sum() == 0:
        raise ValueError("counts sum to zero")
    groups = np.concatenate([np.full(c, i, dtype=int) for i, c in enumerate(counts)])
    return groups

def _complex_to_real(X, mode="split"):
    """
    Convert complex array to real feature matrix.
    X : ndarray (n_samples, n_features) possibly complex dtype
    mode :
      - "split" : stack real and imag -> shape (n_samples, 2*n_features)
      - "magphase" : convert to magnitude and circular descriptors: [mag, cos(phase), sin(phase)] per channel -> shape (n, 3*n_features)
      - "cos_sin_phase" : assume input is phases (angles), map to (cos, sin) per channel -> shape (n, 2*n_features)
      - "identity" (default for real inputs) : return X as float
    """
    X = np.asarray(X)
    if not np.iscomplexobj(X):
        return X.astype(float)
    # else:
    #     raise NotImplementedError("Only 'identity' mode is currently implemented for complex inputs. (uncomment below to see how it works)")
    if mode == "split":
        real = X.real
        imag = X.imag
        return np.hstack([real, imag]).astype(float)
    elif mode == "magphase":
        mag = np.abs(X)
        phase = np.angle(X)
        cosp = np.cos(phase)
        sinp = np.sin(phase)
        return np.hstack([mag, cosp, sinp]).astype(float)
    elif mode == "cos_sin_phase":
        # Useful when X contains phases (angles) only
        phase = np.angle(X)
        cosp = np.cos(phase)
        sinp = np.sin(phase)
        return np.hstack([cosp, sinp]).astype(float)
    else:
        raise ValueError("Unknown complex conversion mode: %s" % mode)


def logistic_l2_cv(train_X, train_y, test_X,
                   samples_per_subject_train=None,
                   cv=5,
                   Cs=(0.001,0.01, 0.1, 1, 10, 100,1000),
                   scoring="accuracy",
                   complex_mode="split",
                   random_state=0,
                   max_iter=10000,
                   return_proba=True):
    """
    L2-regularized logistic regression with CV over C (inverse regularization).
    - train_X: (n_train, d) array (real or complex)
    - train_y: (n_train,) integer labels
    - test_X:  (n_test, d)
    - samples_per_subject_train: optional sequence of ints (counts per subject). If provided we use GroupKFold.
    - Cs: list/iterable of C values to search
    - complex_mode: how to convert complex inputs ("split" is recommended)
    - scoring: scoring metric for CV (see sklearn docs)
    Returns dict with keys: 'pred' (hard labels), 'proba' (probabilities or None), 'best_params_'.
    """
    # --- data conversion (complex -> real) ---
    Xtr = _complex_to_real(train_X, mode=complex_mode)
    Xte = _complex_to_real(test_X, mode=complex_mode)
    ytr = np.asarray(train_y)
    Xtr = np.abs(Xtr)
    Xte = np.abs(Xte)
    # Xtr = Xtr[:1940*1]
    # ytr = ytr[:1940*1]
    # samples_per_subject_train = 1*[1940]

    # --- group-aware CV if counts provided ---
    # groups = _make_groups_from_counts(samples_per_subject_train)
    groups = None
    if groups is not None:
        n_subjects = len(samples_per_subject_train)
        n_splits = min(cv, n_subjects)
        cv_splitter = GroupKFold(n_splits=n_splits)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # class_weights = None
    class_weights = 'balanced'

    # --- pipeline: scaling + logistic ---
    pipe = Pipeline([
        # ("scaler", StandardScaler()),
        # ("pca", PCA(n_components=10, random_state=random_state)),
        ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=max_iter, random_state=random_state, class_weight=class_weights))
    ])

    param_grid = {"clf__C": list(Cs)}

    grid = GridSearchCV(pipe, param_grid, cv=cv_splitter, scoring=scoring, n_jobs=1, refit=True, verbose=1000)
    grid.fit(Xtr, ytr, **({"groups": groups} if groups is not None else {}))

    best = grid.best_estimator_
    y_pred = best.predict(Xte)
    test_proba = best.predict_proba(Xte) if return_proba else None
    train_proba = best.predict_proba(Xtr) if return_proba else None
    # one hot encode
    # train_proba = np.eye(len(np.unique(ytr)))[train_proba]
    # test_proba = np.eye(len(np.unique(ytr)))[test_proba]

    return grid.best_params_, train_proba.T, test_proba.T


def svm_linear_cv(train_X, train_y, test_X,
                  samples_per_subject_train=None,
                  cv=5,
                  Cs=(0.001,0.01, 0.1, 1, 10,100),
                  scoring="accuracy",
                  complex_mode="split",
                  random_state=0,
                  return_proba=True):
    """
    Linear SVM (SVC with linear kernel) with CV over C.
    - For probabilities we set probability=True on SVC (Platt scaling).
    """
    Xtr = _complex_to_real(train_X, mode=complex_mode)
    Xte = _complex_to_real(test_X, mode=complex_mode)
    ytr = np.asarray(train_y)
    # Xtr = Xtr[:1940*3]
    # ytr = ytr[:1940*3]
    # samples_per_subject_train = 3*[1940]

    # groups = _make_groups_from_counts(samples_per_subject_train)
    groups = None
    if groups is not None:
        n_subjects = len(samples_per_subject_train)
        n_splits = min(cv, n_subjects)
        cv_splitter = GroupKFold(n_splits=n_splits)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    class_weights = None
    class_weights = 'balanced'

    pipe = Pipeline([
        # ("scaler", StandardScaler()),
        # ("pca", PCA(n_components=10, random_state=random_state)),
        ("svc", SVC(kernel="linear", probability=False, random_state=random_state, class_weight=class_weights))
    ])
    param_grid = {"svc__C": list(Cs)}

    grid = GridSearchCV(pipe, param_grid, cv=cv_splitter, scoring=scoring, n_jobs=1, refit=True, verbose=1000)
    grid.fit(Xtr, ytr, **({"groups": groups} if groups is not None else {}))

    best = grid.best_estimator_
    train_proba = best.predict(Xtr)
    test_proba = best.predict(Xte)
    # one hot encode
    train_proba = np.eye(len(np.unique(ytr)))[train_proba]
    test_proba = np.eye(len(np.unique(ytr)))[test_proba]
    # test_proba = best.predict_proba(Xte) if return_proba else None
    # train_proba = best.predict_proba(Xtr) if return_proba else None

    return grid.best_params_, train_proba.T, test_proba.T


def svm_rbf_cv(train_X, train_y, test_X,
               samples_per_subject_train=None,
               cv=5,
               Cs=(0.01, 0.1, 1, 10, 100),
               gammas=("scale", 0.01, 0.1, 1, 10),
               scoring="accuracy",
               complex_mode="split",
               random_state=0,
               return_proba=True):
    """
    RBF-kernel SVM with CV on C and gamma.
    - gammas: sequence of gamma values (or 'scale'/'auto')
    """
    Xtr = _complex_to_real(train_X, mode=complex_mode)
    Xte = _complex_to_real(test_X, mode=complex_mode)
    ytr = np.asarray(train_y)

    groups = _make_groups_from_counts(samples_per_subject_train)
    if groups is not None:
        n_subjects = len(samples_per_subject_train)
        n_splits = min(cv, n_subjects)
        cv_splitter = GroupKFold(n_splits=n_splits)
    else:
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    class_weights = 'balanced'

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", probability=False, random_state=random_state, class_weight=class_weights))
    ])
    param_grid = {"svc__C": list(Cs), "svc__gamma": list(gammas)}

    grid = GridSearchCV(pipe, param_grid, cv=cv_splitter, scoring=scoring, n_jobs=-1, refit=True, verbose=1000)
    grid.fit(Xtr, ytr, **({"groups": groups} if groups is not None else {}))

    best = grid.best_estimator_
    # y_pred = best.predict(Xte)
    # test_proba = best.predict_proba(Xte) if return_proba else None
    # train_proba = best.predict_proba(Xtr) if return_proba else None
    train_proba = best.predict(Xtr)
    test_proba = best.predict(Xte)
    # one hot encode
    train_proba = np.eye(len(np.unique(ytr)))[train_proba]
    test_proba = np.eye(len(np.unique(ytr)))[test_proba]

    return grid.best_params_, train_proba.T, test_proba.T
