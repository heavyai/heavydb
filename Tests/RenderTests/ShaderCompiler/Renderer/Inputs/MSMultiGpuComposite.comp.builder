{
  "baseTemplate": "Rendering/MSMultiGpuComposite.comp",
  "operators": [
    {
      "name": "kReplaceFirstTag",
      "string1": "numSamples",
      "string2": "4",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "doMultiSample",
      "string2": "1",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "workgroupSize",
      "string2": "32",
      "required": false
    }
  ]
}