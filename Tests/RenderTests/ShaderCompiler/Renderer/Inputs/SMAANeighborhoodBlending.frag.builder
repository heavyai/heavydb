{
  "baseTemplate": "Rendering/SMAANeighborhoodBlending.frag",
  "operators": [
    {
      "name": "kAddPreamble",
      "string1": "\n#define SMAA_PRESET_HIGH",
      "string2": "",
      "required": false
    },
    {
      "name": "kReplaceFirstTag",
      "string1": "numSamples",
      "string2": "4",
      "required": false
    }
  ]
}