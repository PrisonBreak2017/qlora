from huggingface_hub import HfApi, HfFolder, Repository

# 填写您的 Hugging Face Model Hub 用户名
USERNAME = "michaelwei77"
# 填写您的模型名称
MODEL_NAME = "qlora-llama-7b"

# 初始化 API 和 Repository
api = HfApi()
repo = Repository(f"{USERNAME}/{MODEL_NAME}")

# 将模型文件上传至 Repository
model_path = "output/guanaco7b"  # 指向训练好的模型文件的路径
model_revision = repo.create_revision(title="Initial version", description="First version of the model")
repo.upload_file(model_path, model_revision)

# 将配置文件上传至 Repository
config_path = "output/guanaco7b"  # 指向配置文件的路径
repo.upload_file(config_path, model_revision)

# 将词汇表上传至 Repository
vocab_path = "output/guanaco7b"  # 指向词汇表文件的路径
repo.upload_file(vocab_path, model_revision)

# 提交上传的文件
api.create_repo(token="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", repo_name=f"{USERNAME}/{MODEL_NAME}")

# 完成上传
print(f"模型 {MODEL_NAME} 成功上传至 {USERNAME}/{MODEL_NAME}!")
