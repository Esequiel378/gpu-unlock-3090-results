{ config, lib, pkgs, ... }:
let
  cfg = config.services.gpu-unlock;

  # 1.0.0: the last release with clean RTX 3090 reports. A later commit that
  # reworked the approach broke 3090s once (upstream #20, #23) and was reverted.
  src = pkgs.fetchFromGitHub {
    owner = "Zagitta";
    repo = "gpu-unlock";
    rev = "990a14e9b7dc8eaad8e6925133aa0e8382efe6cf";
    hash = "sha256-mSwqkpNoXJ+f6B4MVhHfYl9u2gFwU07Z+7eAPCwrcho=";
  };
  nv = "610.43.03";
  # hashes from upstream's firmware/artifacts-610.43.03.json; the generators
  # re-check every extracted file against it
  nvSource = pkgs.fetchurl {
    url = "https://download.nvidia.com/XFree86/NVIDIA-kernel-module-source/NVIDIA-kernel-module-source-${nv}.tar.xz";
    sha256 = "7e118923c7a23edc36114d63273a46e3e04e9af98695a42203e7ac2dfe9fc1dc";
  };
  nvUtils = pkgs.fetchurl {
    url = "https://archive.cachyos.org/archive/cachyos-v3/nvidia-utils-${nv}-1-x86_64_v3.pkg.tar.zst";
    sha256 = "e17e7c92eeffa178b0fe1f40eb17a47095f8fc3ce4a77bf115cf5bd5a4df86e9";
  };

  firmware = pkgs.runCommand "gpu-unlock-firmware-${nv}" {
    nativeBuildInputs = [ pkgs.python3 pkgs.zstd ];
  } ''
    mkdir nvsrc
    tar -xf ${nvSource} -C nvsrc
    python3 ${src}/firmware/generate_v67.py ${nvUtils} assets
    python3 ${src}/firmware/extract_boot_assets.py nvsrc/NVIDIA-kernel-module-source-${nv} assets
    install -Dm644 -t $out/lib/firmware/gpu-unlock/${nv} \
      assets/{booter_load_ga102.bin,gsp_ga10x.bin,gsp_boot_ga102.bin,gsp_boot_desc_ga102.bin,v67-ga102-${nv}.bin}
  '';

  kernel = config.boot.kernelPackages.kernel;
  module = pkgs.stdenv.mkDerivation {
    pname = "gpu-unlock";
    version = "1.0.0-${kernel.modDirVersion}";
    inherit src;
    sourceRoot = "source/kernel/gpu_unlock";
    nativeBuildInputs = kernel.moduleBuildDependencies;
    makeFlags = [ "KDIR=${kernel.dev}/lib/modules/${kernel.modDirVersion}/build" ];
    installPhase = ''
      install -Dm644 gpu_unlock.ko $out/lib/modules/${kernel.modDirVersion}/extra/gpu_unlock.ko
    '';
  };
in
{
  options.services.gpu-unlock = {
    enable = lib.mkEnableOption "Zagitta/gpu-unlock, which lifts the GeForce tensor-core throttle on Ampere GPUs at boot";

    targets = lib.mkOption {
      type = lib.types.nonEmptyListOf (lib.types.strMatching "[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\\.[0-7]");
      example = [ "0000:01:00.0" ];
      description = ''
        Full PCI addresses of the cards to unlock, as `lspci -D` prints them.
        No default on purpose: name each card rather than unlocking every one.
      '';
    };

    stockSpecialisation = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Add a `stock` boot entry without the unlock. Rolling back still needs
        a full power-off before booting it: a reboot keeps the card unlocked.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    assertions = [{
      assertion = pkgs.stdenv.hostPlatform.isx86_64;
      message = "services.gpu-unlock: the firmware inputs are x86_64 only";
    }];

    boot.extraModulePackages = [ module ];
    hardware.firmware = [ firmware ];
    # from the initrd so it unlocks before nvidia (loaded later) can bind: the
    # module skips any card that already has a driver. The initrd picks up the
    # firmware through the module's MODULE_FIRMWARE.
    boot.initrd.kernelModules = [ "gpu_unlock" ];
    # resume=0: the resume path does a second reset on a suspended card; a
    # machine that never suspends doesn't need that risk
    boot.kernelParams = [
      "gpu_unlock.targets=${lib.concatStringsSep "," cfg.targets}"
      "gpu_unlock.resume=0"
    ];

    specialisation = lib.mkIf cfg.stockSpecialisation {
      stock.configuration.services.gpu-unlock.enable = lib.mkForce false;
    };
  };
}
