let
  rustOverlay = builtins.fetchTarball "https://github.com/oxalica/rust-overlay/archive/master.tar.gz";
  pkgs = import <nixpkgs> {
    overlays = [(import rustOverlay)];
  };
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      cmake
      fontconfig
      pkg-config
      rust-bin.stable.latest.default
      rust-analyzer
    ];

    RUST_BACKTRACE = 1;
  }
