CREATE SERVER benchmark_s3_csv FOREIGN DATA WRAPPER delimited_file WITH (storage_type='AWS_S3', s3_bucket='benchmark-fsi-dataset', aws_region='us-west-1')
