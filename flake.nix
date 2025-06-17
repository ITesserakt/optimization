{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    fenix.url = "github:nix-community/fenix";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, fenix }: let 
	system = "x86_64-linux";
	pkgs = import nixpkgs { inherit system; };
	toolchain = fenix.packages.${system}.latest.toolchain;
  in {
  	devShells.${system}.default = pkgs.mkShell {
      packages = [ toolchain ];
  	};
  };
}
