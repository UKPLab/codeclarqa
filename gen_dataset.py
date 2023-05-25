import re
import ast
import json
import pickle
from os.path import join

from tqdm import tqdm

from collections import Counter

from sklearn.model_selection import train_test_split


def load_pickle_file(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def load_json_file(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def camel_case_split(com):
    com = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', com))
    for i in range(5):
        com = com.replace('- ', '-')
    com = com.replace(" :", ":")
    com = com.replace("N As", "NAs")
    com = com.replace(" Na N", "NaN")
    # upload
    com = com.split()
    com = ' '.join(com)
    return com


def filter_samples(code_w_preds, k=3):
    # usually we choose dataset with no more than e.g. 3 missing specifications
    return {e: code_w_preds[e] for e in code_w_preds if
            len(code_w_preds[e]["preds"]) - sum(code_w_preds[e]["preds"]) <= k}


def get_all_paths_docs_and_filter_duplicates(code_w_preds):
    # get all paths, docs, and filter duplicated nodes
    all_paths = {}
    all_docs = {}
    new_code_w_preds = {}
    for idx in code_w_preds:
        sample_paths = []

        sample_lines = []
        sample_source_text = []
        sample_docs = []
        sample_nodes = []
        sample_sims = []
        sample_preds = []

        for i, elem in enumerate(code_w_preds[idx]['related_nodes']):
            sample_path = ".".join([elem["path"][0], elem["path"][-1]])

            # filter out duplicated docs
            if sample_path in sample_paths:
                continue
            sample_paths.append(sample_path)

            sample_lines.append(code_w_preds[idx]['lines'][i])
            sample_source_text.append(code_w_preds[idx]['source_text'][i])
            sample_docs.append(code_w_preds[idx]['related_docs'][i])
            sample_nodes.append(code_w_preds[idx]['related_nodes'][i])
            sample_sims.append(code_w_preds[idx]['sims'][i])
            sample_preds.append(code_w_preds[idx]['preds'][i])

        all_paths[idx] = sample_paths
        all_docs[idx] = sample_docs
        new_code_w_preds[idx] = {'lines': sample_lines,
                                 'source_text': sample_source_text,
                                 'related_docs': sample_docs,
                                 'related_nodes': sample_nodes,
                                 'sims': sample_sims,
                                 'preds': sample_preds}

    return new_code_w_preds, all_paths, all_docs


def count_overlapping_nodes(list1, list2):
    count = 0
    for elem in list1:
        if elem in list2:
            count += 1
    return count


def get_set_paths(code_w_preds):
    set_paths = []
    for idx in code_w_preds:
        for elem in code_w_preds[idx]['related_nodes']:
            set_paths.append(".".join([elem["path"][0], elem["path"][-1]]))

    return set_paths


def get_augmented_paths(all_paths):
    try:
        augmented_paths = load_pickle_file(join(data_dir, "final", "similar_paths.pkl"))
        return augmented_paths
    except:
        pass
    augmented_paths = {}
    for idx1 in tqdm(all_paths):
        if len(all_paths[idx1]) == 0:
            augmented_paths[idx1] = []
            continue
        dissimilar_matches = []
        for idx2 in all_paths:

            # filter out the same sample and empty paths
            if idx1 == idx2 or len(all_paths[idx2]) == 0:
                continue

            similar_counts = count_overlapping_nodes(all_paths[idx1], all_paths[idx2])
            # filter out non-similar paths
            if similar_counts / len(all_paths[idx1]) < 0.5:
                continue

            # filter out ones without dissimilar paths
            dissimilar_paths = [e for e in all_paths[idx2] if e not in all_paths[idx1]]
            if len(dissimilar_paths) == 0:
                continue

            # collect dissimilar matches
            dissimilar_matches.extend(dissimilar_paths)

        augmented_paths[idx1] = dissimilar_matches

    pickle.dump(augmented_paths, open("../similar_paths.pkl", "wb"))
    return augmented_paths


def get_most_common(augmented_paths, k=2):  # usually we choose the most common 2 words
    most_common = {}
    for idx in tqdm(augmented_paths):
        if len(augmented_paths[idx]) == 0:
            most_common[idx] = []
            continue
        counter = Counter(augmented_paths[idx])
        counter = [e[0] for e in counter.most_common(k)]
        most_common[idx] = counter

    return most_common


def get_path_2_doc(all_paths):
    path_2_doc = {}
    for e in all_paths:
        for i, path in enumerate(all_paths[e]):
            if path in path_2_doc:
                continue
            path_2_doc[path] = all_docs[e][i]
    return path_2_doc


def get_last_word_path_maps(set_paths):
    path_2_last_word = {e: camel_case_split(e.split(".")[-1]).strip().replace("_", " ").lower().split()[-1] for e in
                        set_paths}
    last_word_2_path = {}
    for path in path_2_last_word:
        last_word = path_2_last_word[path]
        path = path.split(".")[-1]
        if last_word not in last_word_2_path:
            last_word_2_path[last_word] = []
        if path.lower() in [e.lower() for e in last_word_2_path[last_word]]:
            continue
        last_word_2_path[last_word].append(path)

    return path_2_last_word, last_word_2_path


def update_text_and_code_data(text_data, code_data, indices):
    for idx in code_data:
        code_sample = ast.unparse(ast.parse("\n".join(code_data[idx])))
        code_sample = "\n".join([line for line in code_sample.split("\n") if line != ""])
        code_data[idx] = code_sample
    return {e:text_data[e] for e in indices}, {e:code_data[e] for e in indices}


def get_question_type(path):
    method = path.split(".")[-1]
    last_word = ' '.join(camel_case_split(method).strip().split()).replace("_", " ").lower().split()[-1]
    first_word = ' '.join(camel_case_split(method).strip().split()).replace("_", " ").lower().split()[0]

    if last_word in ["model", "classifier", "regressor", "regression", "cv", 
                     "net", "d", "svc", "svm", "svr", "net", "nb", "means", 
                     "analysis", "arima", "ols", "glm"]:

        if first_word in ["get", "load", "select"]:
            return "bipolar", ""

        return "multiple", "model/algorithm"

    elif "plot" in last_word:
        return "multiple", "plot"

    elif len(last_word_2_path[last_word]) >= 2 and last_word in ['frame', 'title', 'feature', 'eval', 'file', 'classifier', 'regressor', 
                                                                 'pool', 'cv', 'fit', 'predict', 'encoder', 'dict', 'plot', 'sum', 'regression', 
                                                                 'split', 'datetime', 'net', 'weights', 'imputer', 'set', 'features', 'format', 
                                                                 'model', 'adam', 'd', 'generator', 'checkpoint', 'tokenizer', 'session', 'array', 
                                                                 'data', 'loss', 'sequences', 'text', 'axis', 'context', 'grid', 'legend', 'subplots', 
                                                                 'params', 'heatmap', 'matrix', 'count', 'graph', 'tagger', 'stemmer', 'tokenize', 'stack', 
                                                                 'svd', 'zeros', 'values', 'duplicates', 'index', 'line', 'mode', 'table', 'csv', 'pickle', 
                                                                 'test', 'transform', 'report', 'distplot', 'mapbox', 'memory', 'style', 'world', 'pandas', 
                                                                 'column', 'seed', 'palette', 'ridge', 'nb', 'transformer', 'vectorizer', 'union', 'pca', 
                                                                 'fold', 'analysis', 'svc', 'svr', 'scaler', 'pipeline', 'score', 'graphviz', 'error', 
                                                                 'importance', 'curve', 'optimizer', 'scope']:
        return "multiple", last_word

    else:
        return "bipolar", ""


def generate_QA(question_type, last_word, path, path_2_doc, aug_data=True):
    doc_only = ":".join(path_2_doc[path].split(":")[1:]).strip()
    if question_type == "bipolar":
        if aug_data == False:
            return "Do you want to call \'{}\' documented as \'{}\'?".format(path, doc_only), "Yes."
        else:
            return "Do you want to call \'{}\' documented as \'{}\'?".format(path, doc_only), "No."
    elif question_type == "multiple":
        if aug_data == False:
            return "Do you want to call anything related to \'{}\'? If yes, which one?".format(last_word), "Yes, I want to call \'{}\'".format(path)
        else:
            return "Do you want to call anything related to \'{}\'? If yes, which one?".format(last_word), "No."
    else:
        raise NameError('You should use input either bipolar or multiple choice questions.')


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
args = parser.parse_args()

data_dir = args.data_dir
text_data = load_json_file(join(data_dir, "origin", "com.json"))
code_data = load_pickle_file(join(data_dir, "origin", "code_data.pickle"))
code_w_preds = load_pickle_file(join(data_dir, "final", "code_w_preds.pkl"))

# only use samples with no more than 5 missing specifications
code_w_preds = filter_samples(code_w_preds, k=5)
# filter out duplicates and get all paths and docs
code_w_preds, all_paths, all_docs = get_all_paths_docs_and_filter_duplicates(code_w_preds)
# get the set of paths
set_paths = get_set_paths(code_w_preds)
# get augmented paths
augmented_paths = get_augmented_paths(all_paths)
# get most common 2 paths as augmentations,
most_common = get_most_common(augmented_paths, k=2)
# get path_2_doc
path_2_doc = get_path_2_doc(all_paths)
# get last word and path maps
path_2_last_word, last_word_2_path = get_last_word_path_maps(set_paths)
# get all indices and update text and code data
indices = [idx for idx in code_w_preds]
text_data, code_data = update_text_and_code_data(text_data, code_data, indices)

# build up the dataset
try:
    dialogue_text_code_data = pickle.load(open(join(data_dir, "final", "all.pkl"), "rb"))
except:
    dialogue_text_code_data = {}
    for idx in tqdm(indices):
        text_sample = text_data[idx]
        code_sample = code_data[idx]
        idx_lines = []

        origin_paths = []
        origin_labels = []
        origin_QAs = []

        prev_doc = [""]
        for i, pred in enumerate(code_w_preds[idx]["preds"]):
            if pred == 0:
                idx_lines.append(code_w_preds[idx]["lines"][i])
                origin_paths.append(all_paths[idx][i])
                question_type, general_word = get_question_type(all_paths[idx][i])
                origin_labels.append((question_type, general_word))
                clarQ, clarA = generate_QA(question_type, general_word, all_paths[idx][i], path_2_doc, aug_data=False)
                origin_QAs.append((clarQ, clarA))

        # build dictionary for the dataset
        dialogue_text_code_data[idx] = {"text": text_sample,
                                        "code": code_sample,
                                        "lines": idx_lines,
                                        "origin_paths": origin_paths,
                                        "origin_labels": origin_labels,
                                        "origin_QAs": origin_QAs,}

    pickle.dump(dialogue_text_code_data, open(join(data_dir, "final", "all.pkl"), "wb"))

# train_dev_test_split
try:
    train_dev_test_split = pickle.load(open(join(data_dir, "final", "train_dev_test_split.pkl"), "rb"))
    train_split = train_dev_test_split[0]
    dev_split = train_dev_test_split[1]
    test_split = train_dev_test_split[2]
except:
    split = train_test_split(indices, shuffle=True, train_size=0.9)
    dev_test_split = train_test_split(split[1], shuffle=True, train_size=0.5)
    train_split = split[0]
    dev_split = dev_test_split[0]
    test_split = dev_test_split[1]
    train_dev_test_split = [train_split, dev_split, test_split]
    pickle.dump(train_dev_test_split, open(join(data_dir, "final", "train_dev_test_split.pkl"), "wb"))

# build code generation dataset
def get_code_gen_data_json(all_data, ids):
    output_data = {"data": []}
    for idx in ids:
        str_idx = str(idx)
        basic_temp = all_data[str_idx]["text"]
        sample_QAs = all_data[str_idx]["origin_QAs"]
        len_QAs = len(sample_QAs)
        for i in range(len_QAs):
            basic_temp += "</s>{}</s>{}".format(sample_QAs[i][0],
                                                sample_QAs[i][1])

        output_data["data"].append({"src": basic_temp,
                                    "tgt": all_data[str_idx]["code"]})

    return output_data


train_gen_data = get_code_gen_data_json(dialogue_text_code_data, train_split)
val_gen_data = get_code_gen_data_json(dialogue_text_code_data, dev_split)
test_gen_data = get_code_gen_data_json(dialogue_text_code_data, test_split)

with open(join(data_dir, 'final', 'gen_code_train.json'), 'w') as outfile:
    outfile.write(json.dumps(train_gen_data))
with open(join(data_dir, 'final', 'gen_code_val.json'), 'w') as outfile:
    outfile.write(json.dumps(val_gen_data))
with open(join(data_dir, 'final', 'gen_code_test.json'), 'w') as outfile:
    outfile.write(json.dumps(test_gen_data))


# build code generation baseline dataset
def get_code_gen_baseline_data_json(all_data, ids):
    output_data = {"data": []}
    for idx in ids:
        str_idx = str(idx)
        output_data["data"].append({"src": all_data[str_idx]["text"],
                                    "tgt": all_data[str_idx]["code"]})

    return output_data


train_gen_baseline_data = get_code_gen_baseline_data_json(dialogue_text_code_data, train_split)
val_gen_baseline_data = get_code_gen_baseline_data_json(dialogue_text_code_data, dev_split)
test_gen_baseline_data = get_code_gen_baseline_data_json(dialogue_text_code_data, test_split)

with open(join(data_dir, 'final', 'gen_code_baseline_train.json'), 'w') as outfile:
    outfile.write(json.dumps(train_gen_baseline_data))
with open(join(data_dir, 'final', 'gen_code_baseline_val.json'), 'w') as outfile:
    outfile.write(json.dumps(val_gen_baseline_data))
with open(join(data_dir, 'final', 'gen_code_baseline_test.json'), 'w') as outfile:
    outfile.write(json.dumps(test_gen_baseline_data))
