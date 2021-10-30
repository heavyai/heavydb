CREATE SERVER benchmark_s3_parquet FOREIGN DATA WRAPPER omnisci_parquet WITH (storage_type='AWS_S3', s3_bucket='benchmark-fsi-dataset', aws_region='us-west-1')
