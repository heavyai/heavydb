def generate(mask, cond_mask, upper_bounds):
  indent_level = 0
  start_iterator_ch = 'i'
  cond_idx = 0
  iterators = []
  loops = ''
  for i in xrange(0, len(upper_bounds)):
    if mask & (1 << i):
      cond_is_true = (cond_mask & (1 << cond_idx)) != 0
      slot_lookup_result = 99 if cond_is_true else -1
      if_cond = indent_level * '  ' + 'if %d >= 0:' % slot_lookup_result
      cond_idx += 1
      iterators.append(slot_lookup_result)
      loops += if_cond + '\n'
    else:
      iterator_ch = chr(ord(start_iterator_ch) + i)
      for_loop = (indent_level * '  ' + 'for %s in xrange(0, %d):' %
        (iterator_ch, upper_bounds[i]))
      iterators.append(iterator_ch)
      loops += for_loop + '\n'
    indent_level += 1
  loops += (indent_level * '  ' + "print '" + ', '.join(['%s' for iterator in iterators])
    + "' % (" + ', '.join([str(iterator) for iterator in iterators]) + ')')
  return loops

if __name__ == '__main__':
  upper_bounds = [5, 3, 9]
  for mask in xrange(0, 1 << len(upper_bounds)):
    mask_bitcount = bin(mask).count("1")
    for cond_mask in xrange(0, 1 << mask_bitcount):
      loops = generate(mask, cond_mask, upper_bounds)
      exec(loops)
