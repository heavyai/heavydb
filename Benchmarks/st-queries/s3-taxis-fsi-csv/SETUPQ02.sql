CREATE SERVER benchmark_s3_csv FOREIGN DATA WRAPPER omnisci_csv WITH (storage_type='AWS_S3', s3_bucket='benchmark-fsi-dataset', aws_region='us-west-1')
