/*
 *  Some cool MapD License
 */
package com.mapd.calcite.parser;

import org.apache.calcite.rel.RelNode;
import org.apache.calcite.rel.externalize.MapDRelJsonWriter;

/**
 *
 * @author michael
 */
public class MapDSerializer {

  public static String toString(final RelNode rel) {
    if (rel == null) {
      return null;
    }
    final MapDRelJsonWriter planWriter = new MapDRelJsonWriter();
    rel.explain(planWriter);
    return planWriter.asString();
  }
}
