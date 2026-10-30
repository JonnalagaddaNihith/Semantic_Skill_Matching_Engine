from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction import DictVectorizer


def make_features(dense, tasks, task_vectorizer, domains_a, domains_b, same_domain, fit=False):
    task_records = [{"task_type": str(t)} for t in tasks]
    domain_a_records = [{"domain": str(d)} for d in domains_a]
    domain_b_records = [{"domain": str(d)} for d in domains_b]

    if fit:
        task_features = task_vectorizer.fit_transform(task_records)
        domain_vectorizer = DictVectorizer(sparse=True)
        domain_a_features = domain_vectorizer.fit_transform(domain_a_records)
        domain_b_features = domain_vectorizer.transform(domain_b_records)
    else:
        task_features = task_vectorizer.transform(task_records)
        domain_vectorizer = task_vectorizer.domain_vectorizer
        domain_a_features = domain_vectorizer.transform(domain_a_records)
        domain_b_features = domain_vectorizer.transform(domain_b_records)

    # Stack all features
    same_domain_matrix = csr_matrix(same_domain.values.reshape(-1, 1))
    return hstack(
        [csr_matrix(dense), task_features, domain_a_features, domain_b_features, same_domain_matrix],
        format="csr",
    ), domain_vectorizer
