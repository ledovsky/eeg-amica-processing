base_path = "C:\\...\test_data\kids";
preprocessed_path = base_path + "\preprocessed";
locactions_path = base_path + "\montage.loc";
amica_tmp_path = base_path + "\amica_tmp";
amica_out_path = base_path + "\amica_raw";

locs = readlocs(convertStringsToChars(locactions_path), 'filetype', 'custom', 'format', {'labels', 'X', 'Y', 'Z'});

fns = dir(preprocessed_path);
fns = fns(~ismember({fns.name}, {'.','..'}));

for i = 1:length(fns)
    disp(preprocessed_path + '\' + fns(i).name);
    sphere_fn = strrep(fns(i).name, '.edf', '_amica_s.csv');
    weights_fn = strrep(fns(i).name, '.edf', '_amica_w.csv');
    
    EEG = pop_biosig(convertStringsToChars(preprocessed_path + '\' + fns(i).name));
    [W,S,mods] = runamica15(EEG.data(:,:), 'max_iter', 100, 'num_models', 1, 'outdir', convertStringsToChars(amica_tmp_path));
    csvwrite(convertStringsToChars(amica_out_path + '\' + sphere_fn), S);
    csvwrite(convertStringsToChars(amica_out_path + '\' + weights_fn), W);
end
