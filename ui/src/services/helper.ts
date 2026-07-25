export const buildSubjectTitle = (subjectId: string): string => {
  return subjectId
      .replace(/^\d+-/, "")
      .replace(/[-_]/g, " ")
      .replace(/\b\w/g, (c) => c.toUpperCase())
}
