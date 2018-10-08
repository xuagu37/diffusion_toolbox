function data = load_nii_data(dir)

nii = load_untouch_nii(dir);
data = double(nii.img);

end
