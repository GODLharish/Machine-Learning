import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import schema_pb2
import os
import tempfile
import subprocess

def download_from_gcs(gcs_path, local_path):
    
    subprocess.run(['gsutil', 'cp', gcs_path, local_path], check=True)

def upload_to_gcs(local_path, gcs_path):
    
    subprocess.run(['gsutil', 'cp', local_path, gcs_path], check=True)

def main():
    # GCS paths
    input_csv = 'gs://loan-eligibility/raw/train.csv'
    output_schema = 'gs://loan-eligibility/schema/schema.pbtxt'
    
    # Create temporary directory for local files
    with tempfile.TemporaryDirectory() as temp_dir:
        local_csv = os.path.join(temp_dir, 'train.csv')
        
        # Download CSV from GCS
        print("Downloading CSV from GCS...")
        download_from_gcs(input_csv, local_csv)
        
        # Infer schema
        print("Inferring schema...")
        schema = tfdv.infer_schema(tfdv.generate_statistics_from_csv(local_csv))
        
        # Save schema locally
        local_schema = os.path.join(temp_dir, 'schema.pbtxt')
        tfdv.write_schema_text(schema, local_schema)
        
        # Upload schema to GCS
        print("Uploading schema to GCS...")
        upload_to_gcs(local_schema, output_schema)
        
        print(f"Schema has been successfully generated and saved to {output_schema}")

if __name__ == '__main__':
    main() 