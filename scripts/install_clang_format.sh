#!/bin/bash

die() {
  echo "$@" 1>&2 ; popd 2>/dev/null; exit 1
}

install_apt_pkg() {
  wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
  add-apt-repository 'deb http://apt.llvm.org/focal/ llvm-toolchain-focal-17 main' &&
  apt-get update -qq && apt-get install -qq -y clang-format-17
}

install_brew_pkg() {
  brew install clang-format@17
}

install_clang_format() {
  if [[ $OSTYPE == linux* ]]; then
    if [ -n "$(command -v apt-get)" ]; then
      install_apt_pkg || die "could not install apt clang format package"
    else
      die "unsupported Linux package management system"
    fi
  elif [[ $OSTYPE == darwin* ]]; then
    if [ -n "$(command -v brew)" ]; then
      install_brew_pkg || die "could not install brew clang format package"
    else
      die "homebrew is not installed!"
    fi
  else
    die "unsupported OS"
  fi
}

run() {
  install_clang_format
}

run
