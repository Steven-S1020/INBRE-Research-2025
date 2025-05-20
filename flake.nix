{
  description = "Flake for {{name}} Project";

  inputs.system-flake.url = "path:/etc/nixos";
  inputs.nixpkgs.follows = "system-flake/nixpkgs";

  outputs =
    { system-flake, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      # Base devShell to use
      base = system-flake.devShells.${system}.dsci;
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        # Extra Packages not included in base devShell
        buildInputs = base.buildInputs ++ [
          # Add Packages Here
        ];
      };
    };
}

