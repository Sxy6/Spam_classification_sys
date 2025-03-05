import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


from dask_ml.model_selection import GridSearchCV as DaskGridSearchCV  # 使用 Dask 的分布式 GridSearch
from dask.distributed import Client  # 连接分布式集群
import multiprocessing



# # 读取数据（假设已经清理并提取了特征）
# data_file = "E:\Study\Graduation\数据集\.venv\output_tokenized.csv"  # 处理后数据文件
# # # 用 Dask 读取 CSV（自动并行化）
# # df = dd.read_csv('output_tokenized.csv')
#
# df = pd.read_csv(data_file)
#
# # 处理数据，假设 'text' 是文本列，'label' 是目标类别（需要根据你的数据调整）
# if 'Message' not in df.columns or 'Spam/Ham' not in df.columns:
#     raise ValueError("CSV 文件中需要 'text' 和 'label' 列")
#
# # 划分数据集（6:2:2）
# X_train, X_temp, y_train, y_temp = train_test_split(df['Message'], df['Spam/Ham'], test_size=0.4, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
#
# print(f"数据集划分完成：训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")
#
#
#
# ### 1️⃣ 训练 贝叶斯分类器 ###
# nb_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),  # 文本特征提取
#     ('nb', MultinomialNB())  # 朴素贝叶斯
# ])
#
# # # 检查 NaN 值
# # print(X_train.isnull().sum())
#
# # 去除 NaN 或填充空值
# X_train = X_train.fillna("")  # 用空字符串替代 NaN
#
# X_val = X_val.fillna("")  # 用空字符串替代 NaN
#
#
# nb_pipeline.fit(X_train, y_train)  # 训练模型
# y_pred_nb = nb_pipeline.predict(X_val)
# nb_acc = accuracy_score(y_val, y_pred_nb)  # 计算准确率
# print(f"贝叶斯分类器 验证集准确率: {nb_acc:.4f}")
#
# # 如果准确率达不到 89%，尝试调整参数（例如 min_df, max_df, ngram_range）
# if nb_acc < 0.89:
#     print("尝试优化 贝叶斯模型 参数...")
#     nb_pipeline.set_params(tfidf__min_df=2, tfidf__ngram_range=(1,2))
#     nb_pipeline.fit(X_train, y_train)
#     y_pred_nb = nb_pipeline.predict(X_val)
#     nb_acc = accuracy_score(y_val, y_pred_nb)
#     print(f"优化后 贝叶斯分类器 验证集准确率: {nb_acc:.4f}")
#
# # # 训练 SVM
# # vectorizer = TfidfVectorizer()
# # X_train = vectorizer.fit_transform(X_train.astype(str))
# # svm_model = SVC(kernel='linear')
# # svm_model.fit(X_train, y_train)

### 2️⃣ 训练 SVM 模型 + 核函数调参 ###
if __name__ == '__main__':
    multiprocessing.freeze_support()  # 添加 freeze_support()

    # --------------------------
    # 启动 Dask 客户端（使用不同端口）
    # --------------------------
    client = Client(n_workers=4, threads_per_worker=1, dashboard_address=':8788')  # 修改默认端口
    print("Dask 集群信息:", client)

    # --------------------------
    # 数据预处理代码保持不变...
    # 读取数据（假设已经清理并提取了特征）
    data_file = "E:\Study\Graduation\数据集\.venv\output_tokenized.csv"  # 处理后数据文件
    # # 用 Dask 读取 CSV（自动并行化）
    # df = dd.read_csv('output_tokenized.csv')

    df = pd.read_csv(data_file)

    # 处理数据，假设 'text' 是文本列，'label' 是目标类别（需要根据你的数据调整）
    if 'Message' not in df.columns or 'Spam/Ham' not in df.columns:
        raise ValueError("CSV 文件中需要 'text' 和 'label' 列")

    # 划分数据集（6:2:2）
    X_train, X_temp, y_train, y_temp = train_test_split(df['Message'], df['Spam/Ham'], test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"数据集划分完成：训练集 {len(X_train)}, 验证集 {len(X_val)}, 测试集 {len(X_test)}")

    ### 1️⃣ 训练 贝叶斯分类器 ###
    nb_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),  # 文本特征提取
        ('nb', MultinomialNB())  # 朴素贝叶斯
    ])

    # # 检查 NaN 值
    # print(X_train.isnull().sum())

    # 去除 NaN 或填充空值
    X_train = X_train.fillna("")  # 用空字符串替代 NaN

    X_val = X_val.fillna("")  # 用空字符串替代 NaN

    nb_pipeline.fit(X_train, y_train)  # 训练模型
    y_pred_nb = nb_pipeline.predict(X_val)
    nb_acc = accuracy_score(y_val, y_pred_nb)  # 计算准确率
    print(f"贝叶斯分类器 验证集准确率: {nb_acc:.4f}")

    # 如果准确率达不到 89%，尝试调整参数（例如 min_df, max_df, ngram_range）
    if nb_acc < 0.89:
        print("尝试优化 贝叶斯模型 参数...")
        nb_pipeline.set_params(tfidf__min_df=2, tfidf__ngram_range=(1, 2))
        nb_pipeline.fit(X_train, y_train)
        y_pred_nb = nb_pipeline.predict(X_val)
        nb_acc = accuracy_score(y_val, y_pred_nb)
        print(f"优化后 贝叶斯分类器 验证集准确率: {nb_acc:.4f}")

    # # 训练 SVM
    # vectorizer = TfidfVectorizer()
    # X_train = vectorizer.fit_transform(X_train.astype(str))
    # svm_model = SVC(kernel='linear')
    # svm_model.fit(X_train, y_train)
    # --------------------------

    # （原数据读取和预处理代码）

    ### 2️⃣ 训练 SVM 模型 ###

    # 确保所有主要逻辑都在 main 块中
    svm_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('svm', SVC())
    ])

    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf', 'poly'],
        'svm__gamma': ['scale', 'auto']
    }

    grid_search = DaskGridSearchCV(
        svm_pipeline,
        param_grid,
        cv=3,
        scoring='accuracy',
        scheduler=client  # 显式绑定到已创建的客户端
    )

    # 分布式训练
    grid_search.fit(X_train, y_train)
    # 最优参数
    best_svm = grid_search.best_estimator_
    y_pred_svm = best_svm.predict(X_val)
    svm_acc = accuracy_score(y_val, y_pred_svm)
    print(f"SVM 最优模型 验证集准确率: {svm_acc:.4f}")
    print(f"最优 SVM 参数: {grid_search.best_params_}")


    X_test = X_test.fillna("")  # 用空字符串替代 NaN


    # 在测试集上评估最终模型
    final_model = nb_pipeline if nb_acc > svm_acc else best_svm
    y_pred_test = final_model.predict(X_test)
    final_acc = accuracy_score(y_test, y_pred_test)
    print(f"最终选定模型 测试集准确率: {final_acc:.4f}")

#未加集群原代码
# svm_pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('svm', SVC())  # SVM 模型
# ])
#
# # SVM 超参数调优（尝试不同核函数）
# param_grid = {
#     'svm__C': [0.1, 1, 10],  # 正则化参数
#     'svm__kernel': ['linear', 'rbf', 'poly'],  # 核函数选择
#     'svm__gamma': ['scale', 'auto']  # 核函数 gamma
# }
#
# grid_search = GridSearchCV(svm_pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # 最优参数
# best_svm = grid_search.best_estimator_
# y_pred_svm = best_svm.predict(X_val)
# svm_acc = accuracy_score(y_val, y_pred_svm)
# print(f"SVM 最优模型 验证集准确率: {svm_acc:.4f}")
# print(f"最优 SVM 参数: {grid_search.best_params_}")
#
# # 在测试集上评估最终模型
# final_model = nb_pipeline if nb_acc > svm_acc else best_svm
# y_pred_test = final_model.predict(X_test)
# final_acc = accuracy_score(y_test, y_pred_test)
# print(f"最终选定模型 测试集准确率: {final_acc:.4f}")
