from google.cloud import storage


def save_model_to_bucket(model_path, bucket_name, object_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(object_name)
    blob.upload_from_filename(model_path)


# ONLY DEBUGGING FUNCTION
def try_to_save_to_bucket(bucket_name):
    # Define the text content you want to write to the file
    text_content = "This is a short text file created in Python."
    file_path = "debugging_text_file.txt"

    # Open the file in write mode and write the content
    with open(file_path, "w") as file:
        file.write(text_content)

    print(
        f"Text file '{file_path}' has been created and saved with content:\n{text_content}"
    )

    save_model_to_bucket(file_path, bucket_name, file_path)
