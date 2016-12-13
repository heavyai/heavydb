Importing Data
==============

In order to do visual analysis on your data, you may first import your
data through Immerse's import wizard.

Follow these steps to import your data:

1. Click "New Dashboard"
2. Click "Import Data"
3. Select or drag-and-drop your file(s) for upload. If multiple files, columns
   must match. At this time only delimiter-separated formats such as CSV and
   TSV are supported.
4. Choose Import Settings as appropriate

   a. Null string. If, rather than having a blank for null cells in your
      upload document, you have substituted strings such as "NULL", enter
      that string in the provided "Null String" field.  This will cause it
      to be treated as a null on upload.
   b. Delimiter type. Delimiters can be auto-detected, or you can force a
      certain delimiter (comma, pipe, etc.).
   c. Quoted string. Indicate whether your string fields are enclosed by
      quotes. Delimiter characters which appear inside the quotes will be
      ignored.

5. If your column headers contain spaces or SQL-reserved words or
   characters (e.g. "year", "/", "#") the importer alters these
   characters to make them safe, and notifies you of what was changed.
6. At Table Preview screen, which presents sample rows of imported data, choose
   the data types of your columns. **Choosing data types correctly is important
   to ensure optimal performance.** The importer makes an educated guess at
   data type based on sampling, but examining selections yourself is strongly
   recommended.
7. Name the table, and click Save Table.
