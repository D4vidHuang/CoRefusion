package com.example.text;

/**
 * Offline-fallback sample so the pipeline runs without network access.
 * Authored for this benchmark (not copied from any repo). Contains several
 * local variables and parameters that each occur at multiple usage sites,
 * which is exactly what build_dataset.py looks for.
 */
public final class StringScanner {

    private StringScanner() {
    }

    /** Count the non-overlapping occurrences of needle inside haystack. */
    public static int countOccurrences(String haystack, String needle) {
        if (haystack == null || needle == null || needle.isEmpty()) {
            return 0;
        }
        int occurrences = 0;
        int cursor = 0;
        while (cursor <= haystack.length() - needle.length()) {
            int matchIndex = haystack.indexOf(needle, cursor);
            if (matchIndex < 0) {
                break;
            }
            occurrences++;
            cursor = matchIndex + needle.length();
        }
        return occurrences;
    }

    /** Collapse runs of whitespace into single spaces and trim the result. */
    public static String normalizeWhitespace(String input) {
        if (input == null) {
            return "";
        }
        StringBuilder builder = new StringBuilder(input.length());
        boolean previousWasSpace = false;
        for (int position = 0; position < input.length(); position++) {
            char current = input.charAt(position);
            if (Character.isWhitespace(current)) {
                if (!previousWasSpace && builder.length() > 0) {
                    builder.append(' ');
                }
                previousWasSpace = true;
            } else {
                builder.append(current);
                previousWasSpace = false;
            }
        }
        int trailing = builder.length();
        while (trailing > 0 && builder.charAt(trailing - 1) == ' ') {
            trailing--;
        }
        return builder.substring(0, trailing);
    }

    /** Split a camelCase identifier into its lowercase word segments. */
    public static java.util.List<String> splitCamelCase(String identifier) {
        java.util.List<String> segments = new java.util.ArrayList<>();
        StringBuilder segment = new StringBuilder();
        for (int index = 0; index < identifier.length(); index++) {
            char letter = identifier.charAt(index);
            if (Character.isUpperCase(letter) && segment.length() > 0) {
                segments.add(segment.toString().toLowerCase());
                segment.setLength(0);
            }
            segment.append(letter);
        }
        if (segment.length() > 0) {
            segments.add(segment.toString().toLowerCase());
        }
        return segments;
    }
}
