{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs @ { self, nixpkgs, flake-utils, rust-overlay, ... }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };
      in {
        devShells.default = with pkgs; mkShell {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" "rust-analyzer" "rust-docs" "rust-src" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))
          ];
        };
        devShells.nightly = with pkgs; mkShell {
          buildInputs = [
            (rust-bin.selectLatestNightlyWith (
              toolchain: toolchain.default.override {
                extensions = [ "miri" "rust-analyzer" "rust-src" ];
              }
            ))
          ];
        };
        devShells.ci = with pkgs; mkShell {
          buildInputs = [
            (rust-bin.stable.latest.minimal.override {
              extensions = [ "clippy" ];
              # Windows CI unfortunately needs to cross-compile from within WSL because Nix doesn't
              # work on Windows.
              targets = [ "x86_64-pc-windows-msvc" ];
            })
            # We use nightly rustfmt features.
            (rust-bin.selectLatestNightlyWith (toolchain: toolchain.rustfmt))
            typos
          ];
        };
        devShells.ci-msrv = let
          manifest = builtins.fromTOML (builtins.readFile ./Cargo.toml);
          msrv = manifest.package.rust-version;
        in with pkgs; mkShell {
          buildInputs = [
            (rust-bin.stable.${msrv}.minimal.override {
              # Windows CI unfortunately needs to cross-compile from within WSL because Nix doesn't
              # work on Windows.
              targets = [ "x86_64-pc-windows-msvc" ];
            })
          ];
        };
      }
    );
}
