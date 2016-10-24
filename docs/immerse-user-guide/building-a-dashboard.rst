Building a Dashboard
=====================

MapD Immerse dashboards can be constructed quickly with a few easy
steps: 1) choosing a database table to visualize; 2) adding charts; 3)
filtering on those charts; 4) saving the dashboard and sharing with
other MapD users.

Connecting to a table
---------------------

After clicking the "New Dashboard" button, select the table which you
would like to analyze.  Single clicking on the table name in the left
pane shows a preview of the table’s columns and data types in the right
pane.  Double clicking on the table name in the left pane connects to
the table; or, the "Connect to Table" button in the right pane can be
clicked to do the same.

 |connect to table.gif|

Adding a chart
--------------

Dimensions
~~~~~~~~~~

Dimensions are the columns in a query which are going to be grouped.
 For example, if I have a table containing each sale of a car in the
United States, and I want to display the number of cars sold per car
manufacturer, the dimension for the query is car manufacturer; i.e. I
will group the rows in the table by car manufacturer.

A query may also have several dimensions, for example the number of cars
sold by car manufacturer, by state.  When a chart has multiple
dimensions, those dimensions are displayed with dimensions separated by
forward slashes (/), so the car manufacturer by state example would be
displayed "Chevrolet / California."  Table Charts display multiple
dimensions in separate table columns.  The order of dimensions can be
altered in the UI by dragging and dropping the dimensions.

Certain chart types have restrictions on the types of dimensions which
can be used.  For example, histograms and line charts can only have
numerical dimensions.  Choropleths may only have dimensions which
reflect geographic regions.  Pointmaps have no dimensions, since they
display geographic points at the latitude/longitude level only.  Number
charts have no dimensions since their purpose is to present only a
single number.

Binned dimensions
^^^^^^^^^^^^^^^^^

When a dimension is set to be a numerical column, Immerse presents
information grouped by each number, unless there are many numbers, in
which case Immerse automatically creates binned ranges of numbers.  For
example, if I have a table with a large number of distinct numbers from
1 to 1 million, a display of 10 bins would be 1 - 100,000, 100,001 -
200,000, etc.  

Immerse’s automatic binning may be manually disabled to force the
display of all of the dimension’s numbers, though depending on expense
of the query, Immerse may display a notification that the unbinned query
is not available.

Measures
~~~~~~~~

Measures typically are calculated fields like SUM, AVERAGE, etc.  These
calculated values are what is plotted to create Immerse’s charts,
allowing the measures of one dimension to be visually compared with
another.

If you select a numerical column as a measure, you can choose to
aggregate that column as AVERAGE, MIN, MAX, or SUM.  If you choose a
string column (i.e. a text column), Immerse automatically aggregates by
COUNT UNIQUE, which returns the count of distinct strings in that
column.

Depending on the chart, anywhere from 1 - 4 measures can be visualized.
 A simple example is Pie chart, which offers 2 measures, Size and Color.
 Whichever column you choose as the size measure will be used to
determine the size of the slices in the pie chart.  Color is an optional
second measure for Pie Chart, allowing another layer of information to
be visualized on the same chart.  For example, I might size the Pie’s
slices by the number of sales opportunities which were closed, and color
them by the average deal value of those opportunities.

A more complex chart like Scatterplot can allow 4 measures to be
visualized concurrently.  Scatterplot shows values on a two dimensional
matrix based on an X measure and a Y measure.  Additionally, the dots
used in the scatterplot can be sized and colored by two additional
measures, for 4 measures in total.

Special behavior with Table Charts, for measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For table charts, which are a non-graphical, row/columns presentation of
raw data, measures can have behavior which is distinct from other chart
types.  If you create a table chart with no dimensions, but only
measures, then no grouping occurs on the data, and you are presented
with raw information at the row level from the database.  If you create
a table chart with  dimensions, the measures act in their usual way, as
an aggregate calculation.  Hence, if you would like to view raw, row
level information from the database without performing any
transformations or calculations, create a table chart with no dimensions
and only measures.

.. _custom_measures:

Custom Measures
^^^^^^^^^^^^^^^

In addition to normal Measures, which perform simple aggregation
calculations on data, Immerse allows you to write Custom Measures, which
can perform arbitrary SQL aggregations, for any MapD supported SQL.
 Consider the following prototypical SQL query:

SELECT column1, COUNT(column1) FROM table WHERE column1=’foo’ GROUP BY
column1

The custom measure can be any aggregate statement which can be
substituted for the term COUNT(column1)  above.

For example, a valid custom measure could be a simple multiplication
calculation highlighted below:

SELECT region, product, SUM(units\*price)  FROM table …

Or you might transform the data using a CASE statement, to knock out
values which don’t meet a certain test, using the custom measure
highlighted below:

SELECT column1, CASE WHEN SUM(column2)>10 THEN SUM(column2) ELSE 0 END
 FROM table GROUP BY column1

Since most Immerse charts require Dimensions, which group the data,
Custom Measures for those charts always need to be written as aggregate
statements, as in the examples above.  However, for charts which do not
require dimensions, such a Point Map or Table, it is possible to write
Custom Measures which are not aggregate statements.

Chart title
~~~~~~~~~~~

When you create a chart, Immerse automatically generates a title for the
chart by using the dimension names and measure names.  If you wish, you
may customize the title of the chart, by going to the chart editing
screen, clicking on the chart title, and typing a title of your own.

Save Chart to Dashboard
~~~~~~~~~~~~~~~~~~~~~~~

Once you have created your chart and have it configured to your wishes,
click the Apply button in the upper right to add the chart to your
dashboard.

--------------

Chart filters
-------------

Immerse lets you easily narrow the data being considered by clicking on
areas of the chart which you want to filter for.  For Line and Histogram
charts, filter by dragging left or right along the chart to select a
range.  For Pointmap charts, filter by zooming in on regions (you can
draw a square on the map while holding down the shift key, or you can
use your mousewheel or trackpad to zoom in).  For all other chart types
simply click on the section of the chart you’re interested in.

 |taxi ride filters.gif|

Inverse filters
~~~~~~~~~~~~~~~

If rather than filtering for certain data, you want to filter out  data,
you can do this by clicking the chart while holding down the Command key
(Mac) or Control key (Windows/Linux).  This feature is available for all
chart types other than line, histogram, pointmap, and number.

Global filters

Immerse allows you to filter data at the dashboard level, constraining
data for all charts on the dashboard.  At the top of the dashboard
screen, click Filters / Add Filter, then select the column you’d like to
filter for.  Depending on the data type of the column, Immerse offers
appropriate ways of filtering for that type.  For string columns for
example, you may search using "Contains," "Equals," and other options.

Custom filters
~~~~~~~~~~~~~~

Moving/resizing charts

To move a chart within a dashboard, simply click on the title bar of the
chart, and drag the chart to the new location.  Any chart which already
exists at the location you’re dragging to will get moved aside.  To
resize the chart, click on the lower right corner of the chart and drag
to change the size.

 |moving charts.gif|

Changing chart types

Chart types can be changed by toggling among the chart icons at the top
of the chart editing screen.  If you have already chosen dimensions and
measures for a chart, Immerse indicates which other chart types are also
capable of displaying that data by highlighting the chart icons in
green.  

 |changing chart types.gif|

Some charts require certain types of dimensions/measures and disallow
others, so as you switch between charts you may see dimensions or
measures become deactivated (grayed-out) if they are not appropriate for
that chart type.  Deactivated fields are discarded once a chart is
saved.

Sharing, Saving and Titling dashboards

To save a dashboard, simply click into the title area on the dashboard
screen, type a title and hit Save.  If you want to share the dashboard,
click the Share icon and then the copy icon to copy the URL to your
clipboard.

 |saving dashboards.gif|

Deleting Dashboards
-------------------

To delete a dashboard, click "Dashboards" in the header to go to the
Dashboards screen, mouse over the Dashboard you’d like to delete, and
click the "X" at the far right side of the Dashboard’s row.

.. |connect to table.gif| image:: ./images/connect-to-table.*
.. |taxi ride filters.gif| image:: ./images/taxi-ride-filters.*
.. |moving charts.gif| image:: ./images/moving-charts.*
.. |changing chart types.gif| image:: ./images/changing-chart-types.*
.. |saving dashboards.gif| image:: ./images/saving-dashboards.*
