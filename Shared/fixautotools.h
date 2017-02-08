/*
 * Removes some #defines that should be internal but sometimes sneak in from
 * autotools-based projects.
 */

#ifdef PACKAGE_BUGREPORT
#undef PACKAGE_BUGREPORT
#endif

#ifdef PACKAGE_NAME
#undef PACKAGE_NAME
#endif

#ifdef PACKAGE_STRING
#undef PACKAGE_STRING
#endif

#ifdef PACKAGE_TARNAME
#undef PACKAGE_TARNAME
#endif

#ifdef PACKAGE_URL
#undef PACKAGE_URL
#endif

#ifdef PACKAGE_VERSION
#undef PACKAGE_VERSION
#endif
