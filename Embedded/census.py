import dbe
import ctypes
import pandas
import pyarrow
import numpy
ctypes._dlopen('libDBEngine.so', ctypes.RTLD_GLOBAL)

obj = dbe.PyDbEngine('data', 5555)
#6279)


print('DROP...........................................................')
obj.executeDDL("DROP TABLE census")


print('CREATE...........................................................')
obj.executeDDL('CREATE TABLE census (YEAR0 bigint, DATANUM bigint,\
 SERIAL bigint, CBSERIAL double, HHWT bigint, CPI99 double,\
 GQ bigint, QGQ double, PERNUM bigint, PERWT bigint,\
 SEX bigint, AGE bigint, EDUC bigint, EDUCD bigint,\
 INCTOT bigint, SEX_HEAD double,\
 SEX_MOM double, SEX_POP double,\
 SEX_SP double, SEX_MOM2 double,\
 SEX_POP2 double, AGE_HEAD double,\
 AGE_MOM double, AGE_POP double,\
 AGE_SP double, AGE_MOM2 double,\
 AGE_POP2 double,\
 EDUC_HEAD double,\
 EDUC_MOM double,\
 EDUC_POP double,\
 EDUC_SP double,\
 EDUC_MOM2 double,\
 EDUC_POP2 double,\
 EDUCD_HEAD double,\
 EDUCD_MOM double,\
 EDUCD_POP double,\
 EDUCD_SP double,\
 EDUCD_MOM2 double,\
 EDUCD_POP2 double,\
 INCTOT_HEAD double,\
 INCTOT_MOM double,\
 INCTOT_POP double,\
 INCTOT_SP double,\
 INCTOT_MOM2 double,\
 INCTOT_POP2 double)\
 WITH (fragment_size=1000000)')

#print('RESULT::::::')
#sch = obj.get_table_details('census')
pdf = pandas.DataFrame(
    [
        (
            x.name,
            x.type,
            x.precision,
            x.scale,
            x.comp_param,
            x.encoding,
        )
        for x in obj.get_table_details('census')
    ],
    columns=[
        'column_name',
        'type',
        'precision',
        'scale',
        'comp_param',
        'encoding',
    ],
)
#print(sch)
#print(pdf)

print('IMPORT...........................................................')

obj.executeDDL("COPY census FROM '/localdisk/benchmark_datasets/census/ipums_education2income_1970-2010.csv.gz'  WITH (header='True',quote='\"',quoted='False',delimiter=',')")

print('SELECT...........................................................')
#cur0=obj.executeDML("SELECT * FROM census LIMIT 500")
#print('PROBA:')
#proba_df=obj.select_df("SELECT * FROM census LIMIT 500")
#print('PROBA DF:')
#print(proba_df)
#cur0.showRows(5)

cur=obj.executeDML("SELECT CAST(CASE WHEN YEAR0 IS NOT NULL THEN YEAR0 ELSE -1 END AS DOUBLE) AS YEAR0,\
       CAST(CASE WHEN DATANUM IS NOT NULL THEN DATANUM ELSE -1 END AS DOUBLE) AS DATANUM,\
       CAST(CASE WHEN SERIAL IS NOT NULL THEN SERIAL ELSE -1 END AS DOUBLE) AS SERIAL,\
       CASE WHEN CBSERIAL IS NOT NULL THEN CBSERIAL ELSE -1 END AS CBSERIAL,\
       CAST(CASE WHEN HHWT IS NOT NULL THEN HHWT ELSE -1 END AS DOUBLE) AS HHWT,\
       CASE WHEN CPI99 IS NOT NULL THEN CPI99 ELSE -1 END AS CPI99,\
       CAST(CASE WHEN GQ IS NOT NULL THEN GQ ELSE -1 END AS DOUBLE) AS GQ,\
       CAST(CASE WHEN PERNUM IS NOT NULL THEN PERNUM ELSE -1 END AS DOUBLE) AS PERNUM,\
       CAST(CASE WHEN SEX IS NOT NULL THEN SEX ELSE -1 END AS DOUBLE) AS SEX,\
       CAST(CASE WHEN AGE IS NOT NULL THEN AGE ELSE -1 END AS DOUBLE) AS AGE,\
       CASE WHEN INCTOT IS NOT NULL THEN INCTOT ELSE -1 END AS INCTOT,\
       CAST(CASE WHEN EDUC IS NOT NULL THEN EDUC ELSE -1 END AS DOUBLE) AS EDUC,\
       CAST(CASE WHEN EDUCD IS NOT NULL THEN EDUCD ELSE -1 END AS DOUBLE) AS EDUCD,\
       CASE WHEN EDUC_HEAD IS NOT NULL THEN EDUC_HEAD ELSE -1 END AS EDUC_HEAD,\
       CASE WHEN EDUC_POP IS NOT NULL THEN EDUC_POP ELSE -1 END AS EDUC_POP,\
       CASE WHEN EDUC_MOM IS NOT NULL THEN EDUC_MOM ELSE -1 END AS EDUC_MOM,\
       CASE WHEN EDUCD_MOM2 IS NOT NULL THEN EDUCD_MOM2 ELSE -1 END AS EDUCD_MOM2,\
       CASE WHEN EDUCD_POP2 IS NOT NULL THEN EDUCD_POP2 ELSE -1 END AS EDUCD_POP2,\
       CASE WHEN INCTOT_MOM IS NOT NULL THEN INCTOT_MOM ELSE -1 END AS INCTOT_MOM,\
       CASE WHEN INCTOT_POP IS NOT NULL THEN INCTOT_POP ELSE -1 END AS INCTOT_POP,\
       CASE WHEN INCTOT_MOM2 IS NOT NULL THEN INCTOT_MOM2 ELSE -1 END AS INCTOT_MOM2,\
       CASE WHEN INCTOT_POP2 IS NOT NULL THEN INCTOT_POP2 ELSE -1 END AS INCTOT_POP2,\
       CASE WHEN INCTOT_HEAD IS NOT NULL THEN INCTOT_HEAD ELSE -1 END AS INCTOT_HEAD,\
       CASE WHEN SEX_HEAD IS NOT NULL THEN SEX_HEAD ELSE -1 END AS SEX_HEAD \
FROM (\
  SELECT YEAR0, DATANUM, SERIAL, CBSERIAL, HHWT, CPI99, GQ,\
         PERNUM, SEX, AGE, INCTOT * CPI99 AS INCTOT, EDUC,\
         EDUCD, EDUC_HEAD, EDUC_POP, EDUC_MOM, EDUCD_MOM2,\
         EDUCD_POP2, INCTOT_MOM, INCTOT_POP, INCTOT_MOM2,\
         INCTOT_POP2, INCTOT_HEAD, SEX_HEAD\
  FROM census\
  WHERE (INCTOT <> 9999999) AND (EDUC IS NOT NULL) AND (EDUCD IS NOT NULL)\
) t0")

print('Rows in ResultSet: ', cur.rowCount())
#cur.showRows(15)
#cur.showRows(0)

def test_write_batch(arr):
    print('WRITE SCHEMA TO RB..........................................................')
    ##sink = pyarrow.BufferOutputStream()
    writer = pyarrow.RecordBatchFileWriter('/localdisk5/gal/intel_go/omniscidb/Embedded/out1.arrow', arr.schema)
    print('writer: ', writer)
    ##writer = pyarrow.ipc.new_stream(sink, arr.schema)

#deref(batch.batch)
    print('WRITE BATCH')
    writer.write_batch(arr)
    print('close stream')
    writer.close()

def test_record_batch():
    print('TO ARROW...........................................................')
    arr = cur.getArrowRecordBatch()
    print(type(arr))
    print(dir(arr))
    print('Cols in pyarrow RecordBatch: ', arr.num_columns)
    print('Rows in pyarrow RecordBatch: ', arr.num_rows)
    print(arr.schema)
    print('Validate')
    print(arr.validate(full=True))
    print(arr)
    print('nbytes=', arr.nbytes)
    print('pyarrow RecordBatch batch:')
    #col=arr.column(1)
    #print(col)
    #print(type(col))
    #print(dir(col))
    #print('---------------------------------------------batch')
    #print(type(arr.batch))
    #print(arr.batch)
    #test_write_batch(arr)

    print('TO TABLE...........................................................')
    #arr.to_pydict()
    tbl1=pyarrow.Table.from_batches([arr])
    print(type(tbl1))
    print(dir(tbl1))
    print(tbl1)
    print('nbytes=', tbl1.nbytes)
    print('...TO PANDAS...........................................................')
    #res=arr.to_pandas()
    res=tbl1.to_pandas()
    print(type(res))
    print(dir(res))
    print(list(res.columns.values))
    print(res)
    print('nbytes=', arr.nbytes)
test_record_batch()



