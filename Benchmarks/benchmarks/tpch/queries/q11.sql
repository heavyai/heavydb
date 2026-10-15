select ps_partkey, sum(ps_supplycost * ps_availqty) as ps_value
from partsupp, supplier, nation
where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY'
group by ps_partkey
having sum(ps_supplycost * ps_availqty) > (
  select sum(ps_supplycost * ps_availqty) * 0.0001 / ${SF}
  from partsupp, supplier, nation
  where ps_suppkey = s_suppkey and s_nationkey = n_nationkey and n_name = 'GERMANY')
order by ps_value desc
