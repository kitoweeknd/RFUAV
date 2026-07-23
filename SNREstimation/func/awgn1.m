%  Add white Gaussian noise to reach a target SNR (dB).
%  signalPower is the 3rd argument (used to be hardcoded to 27); it defaults to
%  27 when not supplied, so 2-argument callers keep their old behaviour.
%  NOTE: the file must be named awgn1.m so MATLAB can resolve awgn1(...); the
%  previous name (_AWGN.m) meant callers passing 3 args failed to resolve.
function [noisy_data] = awgn1(dataIQ,targetSNR_dB,signalPower)
    if nargin < 3
        signalPower = 27;
    end
    targetSNR = 10^(targetSNR_dB / 10);
    % Calculate the required noise power
    noisePower = signalPower / targetSNR;

    % generate noise
    noise = sqrt(noisePower/2) .* (randn(size(dataIQ)) + 1j * randn(size(dataIQ)));
    noisy_data = dataIQ + noise;
