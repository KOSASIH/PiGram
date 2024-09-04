import unittest
from blockchain import Blockchain, Block

class TestBlockchain(unittest.TestCase):
    def setUp(self):
        self.blockchain = Blockchain()

    def test_create_genesis_block(self):
        self.assertEqual(len(self.blockchain.chain), 1)
        self.assertEqual(self.blockchain.chain[0].index, 0)
        self.assertEqual(self.blockchain.chain[0].previous_hash, "0")

    def test_add_block(self):
        data = "Hello, World!"
        self.blockchain.add_block(data)
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(self.blockchain.chain[1].index, 1)
        self.assertEqual(self.blockchain.chain[1].data, data)

    def test_validate_chain(self):
        self.assertTrue(self.blockchain.validate_chain())
        self.blockchain.chain[1].data = "Tampered data"
        self.assertFalse(self.blockchain.validate_chain())

    def test_mine_block(self):
        data = "Mining data"
        self.blockchain.mine_block(data)
        self.assertEqual(len(self.blockchain.chain), 3)
        self.assertEqual(self.blockchain.chain[2].index, 2)
        self.assertEqual(self.blockchain.chain[2].data, data)

if __name__ == "__main__":
    unittest.main()
