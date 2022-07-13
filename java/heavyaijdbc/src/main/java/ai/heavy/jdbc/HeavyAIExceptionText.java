package ai.heavy.jdbc;

public class HeavyAIExceptionText {
  static String getExceptionDetail(Exception ex) {
    if (ex.getStackTrace().length < 1) {
      return "Error in stack trace processing";
    }
    StackTraceElement sE = ex.getStackTrace()[0];
    return "[" + sE.getFileName() + ":" + sE.getMethodName() + ":" + sE.getLineNumber()
            + ":" + ex.toString() + "]";
  }
}
