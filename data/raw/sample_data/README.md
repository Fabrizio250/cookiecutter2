# Sample Data description
This folder contains some sample data that can be used for testing purposes.
Are present two subjects:
- OAS30001 (Health)
- OAS30271 (AD demented)

For both the subjects there are 3 different files, each file follows the same naming convention
```<subject_code>_MR_d<days_after_entry>_<type>.<ext>```

### Types description
- brain: is the T1w scan already processed with freesurfer, in other words contains the brain volume only
- brainmask: the scan brain mask, keep in mind that the mask values are not binary
- T1w: the raw mri scan