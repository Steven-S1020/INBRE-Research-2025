{
  description = "Python flake with common packages";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      pythonEnv = pkgs.python3.withPackages (
        ps: with ps; [
          pkgs.python313
          marimo
          matplotlib
          numpy
          pandas
          pip
          python-lsp-server
          seaborn
          scipy
          sympy
        ]
      );
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [ pythonEnv ];
      };
    };
}
